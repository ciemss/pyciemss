import collections
import functools
import json
import operator
import os
from typing import Dict, List, Optional, Union, OrderedDict

import networkx
import numpy
import torch
import pyro

import mira
import mira.modeling
import mira.modeling.petri
import mira.metamodel
import mira.sources
import mira.sources.petri

import bisect

from torchdiffeq import odeint

from pyciemss.interfaces import DynamicalSystem

from pyciemss.PetriNetODE.events import Event, StaticEvent, StartEvent, ObservationEvent, LoggingEvent, StaticParameterInterventionEvent

Time = Union[float, torch.tensor]
State = tuple[torch.tensor]
StateDeriv = tuple[torch.tensor]
Solution = Dict[str, torch.tensor]

class PetriNetODESystem(DynamicalSystem):
    '''
    Base class for ordinary differential equations models in PyCIEMSS.
    '''

    def __init__(self):
        super().__init__()
        # The order of the variables in the state vector used in the `deriv` method.
        self.var_order = self.create_var_order()

        self.reset()

    def reset(self) -> None:
        '''
        Resets the model to its initial state.
        '''
        self._observation_var_names = []
        self._static_events = []
        self._dynamic_stop_events = []
        self._observation_indices = {}
        self._observation_values = {}
        self._observation_indices_and_values_are_set_up = False

    def create_var_order(self) -> dict[str, int]:
        '''
        Returns the order of the variables in the state vector used in the `deriv` method.
        '''
        raise NotImplementedError

    def load_events(self, events: List[Event]) -> None:
        '''
        Loads a list of events into the model.
        '''
        for event in events:
            self.load_event(event)

    def load_event(self, event: Event) -> None:
        '''
        Loads an event into the model.
        '''
        # Execute specializations of the `_load_event` method, dispatched on the type of event.
        self._load_event(event)

        if isinstance(event, StaticEvent):
            # If the event is a static event, then we need to set up the observation indices and values again.
            # We'll do this in the `forward` method if necessary.
            self._observation_indices_and_values_are_set_up = False
            bisect.insort(self._static_events, event)
        else:
            # If the event is a dynamic event, then we need to add it to the list of dynamic events.
            raise NotImplementedError
    
    @functools.singledispatchmethod
    def _load_event(self, event:Event) -> None:
        '''
        Loads an event into the model.
        '''
        pass

    @_load_event.register
    def _load_event_observation_event(self, event: ObservationEvent) -> None:
        # Add the variable names to the list of observation variable names if they are not already there.
        for var_name in event.observation.keys():
            if var_name not in self._observation_var_names:
                self._observation_var_names.append(var_name)

    def _setup_observation_indices_and_values(self):
        '''
        Set up the observation indices and observation values.
        '''
        if not self._observation_indices_and_values_are_set_up:
            self._observation_indices = {}
            self._observation_values = {}
            for var_name in self._observation_var_names:
                self._observation_indices[var_name] = [i for i, event in enumerate(self._static_events) if isinstance(event, ObservationEvent) and var_name in event.observation.keys()]
                self._observation_values[var_name] = torch.stack([self._static_events[i].observation[var_name] for i in self._observation_indices[var_name]])

            self._observation_indices_and_values_are_set_up = True

    def remove_start_event(self) -> None:
        '''
        Remove the start event from the model.
        '''
        self._remove_static_events(StartEvent)

    def remove_observation_events(self) -> None:
        '''
        Remove all observation events from the model.
        '''
        self._observation_var_names = []
        self._remove_static_events(ObservationEvent)

    def remove_logging_events(self) -> None:
        '''
        Remove all logging events from the model.
        '''
        self._remove_static_events(LoggingEvent)

    def remove_static_parameter_intervention_events(self) -> None:
        '''
        Remove all static parameter intervention events from the model.
        '''
        self._remove_static_events(StaticParameterInterventionEvent)

    def _remove_static_events(self, event_class) -> None:
        '''
        Remove all static events of Type `event_class` from the model.
        '''
        self._static_events = [event for event in self._static_events if not isinstance(event, event_class)]
        self._observation_indices_and_values_are_set_up = False

    def deriv(self, t: Time, state: State) -> StateDeriv:
        '''
        Returns a derivate of `state` with respect to `t`.
        '''
        raise NotImplementedError

    @pyro.nn.pyro_method
    def param_prior(self) -> None:
        '''
        Inplace method defining the prior distribution over model parameters.
        All random variables must be defined using `pyro.sample` or `PyroSample` methods.
        '''
        raise NotImplementedError

    @pyro.nn.pyro_method
    def observation_model(self, solution: Dict[str, torch.Tensor], var_name: str) -> None:
        '''
        Conditional distribution of observations given true state trajectory.
        All random variables must be defined using `pyro.sample` or `PyroSample` methods.
        This needs to be called once for each `var_name` in the set of observed variables.
        '''
        raise NotImplementedError
    
    def static_parameter_intervention(self, parameter: str, value: torch.Tensor):
        '''
        Inplace method defining how interventions are applied to modify the model parameters.
        '''
        raise NotImplementedError

    def forward(self, method="dopri5", **kwargs) -> Solution:
        '''
        Joint distribution over model parameters, trajectories, and noisy observations.
        '''
        # Setup the memoized observation indices and values
        self._setup_observation_indices_and_values()

        # Sample parameters from the prior
        self.param_prior()

        # Check that the start event is the first event
        assert isinstance(self._static_events[0], StartEvent)

        # Load initial state
        initial_state = tuple(self._static_events[0].initial_state[v] for v in self.var_order.keys())

        # Get tspan from static events
        tspan = torch.tensor([e.time for e in self._static_events])

        solutions = [tuple(s.reshape(-1) for s in initial_state)]

        # Find the indices of the static intervention events
        bound_indices = [0] + [i for i, event in enumerate(self._static_events) if isinstance(event, StaticParameterInterventionEvent)] + [len(self._static_events)]
        bound_pairs = zip(bound_indices[:-1], bound_indices[1:])

        # Iterate through the static intervention events, running the ODE solver in between each.
        for (start, stop) in bound_pairs:

            if isinstance(self._static_events[start], StaticParameterInterventionEvent):
                # Apply the intervention
                self.static_parameter_intervention(self._static_events[start].parameter, self._static_events[start].value)

            # Construct a tspan between the current time and the next static intervention event
            local_tspan = tspan[start:stop+1]

            # Simulate from ODE with the new local tspan
            local_solution = odeint(self.deriv, initial_state, local_tspan, method=method)

            # Add the solution to the solutions list.
            solutions.append(tuple(s[1:] for s in local_solution))

            # update the initial_state
            initial_state = tuple(s[-1] for s in local_solution)

        # Concatenate the solutions
        solution = tuple(torch.cat(s) for s in zip(*solutions))
        solution = {v: solution[i] for i, v in enumerate(self.var_order.keys())}

        # Compute likelihoods for observations
        for var_name in self._observation_var_names:
            observation_indices = self._observation_indices[var_name]
            observation_values = self._observation_values[var_name]
            filtered_solution = {v: solution[observation_indices] for v, solution in solution.items()}
            with pyro.condition(data={var_name: observation_values}):
                self.observation_model(filtered_solution, var_name)

        # Log the solution
        logging_indices = [i for i, event in enumerate(self._static_events) if isinstance(event, LoggingEvent)]

        # Return the logged solution wrapped in a pyro.deterministic call to ensure it is in the trace
        logged_solution = {v: pyro.deterministic(f"{v}_sol", solution[logging_indices]) for v, solution in solution.items()}

        return logged_solution

@functools.singledispatch
def get_name(obj) -> str:
    """
    Function to get a string-valued name for a MIRA object for use in a Pyro model.

    Guaranteed to be human-readable and unique for Variables,
    and merely unique for everything else.
    """
    raise NotImplementedError


@get_name.register
def _get_name_str(name: str) -> str:
    return name


@get_name.register
def _get_name_variable(var: mira.modeling.Variable) -> str:
    return var.data["name"]


@get_name.register
def _get_name_transition(trans: mira.modeling.Transition) -> str:
    return f"trans_{trans.key}"


@get_name.register
def _get_name_modelparameter(param: mira.modeling.ModelParameter) -> str:
    return f"{param.key[-1]}_{param.key}"


class MiraPetriNetODESystem(PetriNetODESystem):
    """
    Create an ODE system from a petri-net specification.
    """
    def __init__(self, G: mira.modeling.Model):
        self.G = G
        super().__init__()

        for param_info in self.G.parameters.values():
            param_name = get_name(param_info)

            param_value = param_info.value
            if param_value is None:  # TODO remove this placeholder when MIRA is updated
                param_value = torch.nn.Parameter(torch.tensor(0.1))

            if isinstance(param_value, torch.nn.Parameter):
                setattr(self, param_name, pyro.nn.PyroParam(param_value))
            elif isinstance(param_value, pyro.distributions.Distribution):
                setattr(self, param_name, pyro.nn.PyroSample(param_value))
            elif isinstance(param_value, (int, float, numpy.ndarray, torch.Tensor)):
                self.register_buffer(param_name, torch.as_tensor(param_value))
            else:
                raise TypeError(f"Unknown parameter type: {type(param_value)}")

        if all(var.data.get("initial_value", None) is not None for var in self.var_order.values()):
            for var in self.var_order.values():
                self.register_buffer(
                    f"default_initial_state_{get_name(var)}",
                    torch.as_tensor(var.data["initial_value"])
                )

            self.default_initial_state = tuple(
                getattr(self, f"default_initial_state_{get_name(var)}", None)
                for var in self.var_order.values()
            )

    def create_var_order(self) -> dict[str, int]:
        '''
        Returns the order of the variables in the state vector used in the `deriv` method. 
        Specialization of the base class method using the Mira graph object.
        '''
        return collections.OrderedDict(
            (get_name(var), var) for var in sorted(self.G.variables.values(), key=get_name)
        )

    @functools.singledispatchmethod
    @classmethod
    def from_mira(cls, model: mira.modeling.Model) -> "MiraPetriNetODESystem":
        return cls(model)

    @from_mira.register(mira.metamodel.TemplateModel)
    @classmethod
    def _from_template_model(cls, model_template: mira.metamodel.TemplateModel):
        return cls.from_mira(mira.modeling.Model(model_template))

    @from_mira.register(dict)
    @classmethod
    def _from_json(cls, model_json: dict):
        return cls.from_mira(mira.sources.petri.template_model_from_petri_json(model_json))

    @from_mira.register(str)
    @classmethod
    def _from_json_file(cls, model_json_path: str):
        if not os.path.exists(model_json_path):
            raise ValueError(f"Model file not found: {model_json_path}")
        with open(model_json_path, "r") as f:
            return cls.from_mira(json.load(f))

    def to_networkx(self) -> networkx.MultiDiGraph:
        from pyciemss.utils.petri_utils import load
        return load(mira.modeling.petri.PetriNetModel(self.G).to_json())

    @pyro.nn.pyro_method
    def param_prior(self):
        for param_info in self.G.parameters.values():
            getattr(self, get_name(param_info))

    @pyro.nn.pyro_method
    def deriv(self, t: Time, state: State) -> StateDeriv:
        states = {k: state[i] for i, k in enumerate(self.var_order.values())}
        derivs = {k: 0. for k in states}

        population_size = sum(states.values())

        for transition in self.G.transitions.values():
            flux = getattr(self, get_name(transition.rate)) * functools.reduce(
                operator.mul, [states[k] for k in transition.consumed], 1
            )
            if len(transition.control) > 0:
                flux = flux * sum([states[k] for k in transition.control]) / population_size

            for c in transition.consumed:
                derivs[c] -= flux
            for p in transition.produced:
                derivs[p] += flux

        return tuple(derivs[v] for v in self.var_order.values())
    
    @pyro.nn.pyro_method
    def observation_model(self, solution: Solution, var_name: str) -> None:
        # Default implementation just records the observation, with no randomness.
        pyro.deterministic(var_name, solution[var_name])

    def static_parameter_intervention(self, parameter: str, value: torch.Tensor):
        setattr(self, get_name(self.G.parameters[parameter]), value)

class ScaledBetaNoisePetriNetODESystem(MiraPetriNetODESystem):
    '''
    This is a wrapper around PetriNetODESystem that adds Beta noise to the ODE system.
    Additionally, this wrapper adds a uniform prior on the model parameters.
    '''
    def __init__(self, G: mira.modeling.Model, pseudocount: float = 1):

        for param_info in G.parameters.values():
            param_value = param_info.value
            if param_value is None:
                param_info.value = pyro.distributions.Uniform(0.0, 1.0)
            elif isinstance(param_value, (int, float)):
                param_info.value = pyro.distributions.Uniform(max(0.9 * param_value, 0.0), 1.1 * param_value)

        super().__init__(G)
        self.register_buffer("pseudocount", torch.as_tensor(pseudocount))

    def __repr__(self):
        par_string = ",\n\t".join([f"{get_name(p)} = {p.value}" for p in self.G.parameters.values()])
        count_string = f"pseudocount = {self.pseudocount}"
        return f"{self.__class__.__name__}(\n\t{par_string},\n\t{count_string}\n)"

    @pyro.nn.pyro_method
    def observation_model(self, solution: Solution, var_name: str) -> None:
        mean = solution[var_name]
        pseudocount = self.pseudocount
        # TODO: Get `max` from the initial state
        pyro.sample(var_name, ScaledBeta(mean, max, pseudocount).to_event(1))

from torch.distributions import constraints
from torch.distributions import TransformedDistribution
from torch.distributions.transforms import AffineTransform
class ScaledBeta(TransformedDistribution):
    r"""
    Creates a scaled beta distribution parameterized by
    :attr:`mean`, :attr:`max`, and :attr:`pseudocount` where::
        scaled_mean = mean / max
        X ~ Beta(scaled_mean * pseudocount, (1 - scaled_mean) * pseudocount)
        Y = X * max ~ ScaledBeta(mean, max, pseudocount)
    Args:
        mean (float or Tensor): mean of the distribution
        max (float or Tensor): maximum value of the distribution
        pseudocount (float or Tensor): pseudocount for the distribution ( == a + b in the underlying Beta distribution)
    """
    arg_constraints = {'mean': constraints.interval(0, self._max), 'max': constraints.positive, 'pseudocount': constraints.positive}
    # TODO: Fix constraints
    # We really want 0 <= mean <= max and support = [0, max]. Can we express that?
    support = constraints.interval(0, self._max)
    has_rsample = True

    def __init__(self, _mean, _max, pseudocount, validate_args=None):
        self._mean = _mean
        self._max = _max
        self._pseudocount = pseudocount
        scaled_mean = self._mean / self._max
        self._scaled_mean = scaled_mean
        base_dist = pyro.distributions.Beta( scaled_mean * pseudocount, (1 - scaled_mean) * pseudocount, validate_args=validate_args)
        super().__init__(base_dist, AffineTransform(0, _max), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ScaledBeta, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def mean(self):
        return self.base_dist.mean() * self.max

    @property
    def max(self):
        return self._max


    @property
    def variance(self):
        return self.base_dist.variance() * self._max ** 2
