import collections
import functools
import json
import operator
import os
import warnings
from typing import Dict, List, Optional, Union, OrderedDict
import requests
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
import mira.sources.askenet.petrinet as petrinet
import mira.sources.askenet.regnet as regnet
from pyciemss.utils.distributions import ScaledBeta
from mira.metamodel.ops import aggregate_parameters

import bisect

from torchdiffeq import odeint

from pyciemss.interfaces import DynamicalSystem

from pyciemss.PetriNetODE.events import (Event, StaticEvent, StartEvent, ObservationEvent,
                                         LoggingEvent, StaticParameterInterventionEvent)

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
        self.total_population = None

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
            # If the event is a start event, we need to assign
            # self.total_population to the sum of the initial populations.
            if isinstance(event, StartEvent):
                self.total_population = sum(event.initial_state.values())

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
                self._observation_indices[var_name] = [i for i, event in enumerate(self._static_events)
                                                       if isinstance(event, ObservationEvent)
                                                       and var_name in event.observation.keys()]
                self._observation_values[var_name] = torch.stack([self._static_events[i].observation[var_name]
                                                                  for i in self._observation_indices[var_name]])

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

    def setup_before_solve(self) -> None:
        '''
        Inplace method for setting up the model.
        '''
        self._setup_observation_indices_and_values()

    @pyro.nn.pyro_method
    def get_solution(self, method="dopri5") -> Solution:
        # Check that the start event is the first event
        assert isinstance(self._static_events[0], StartEvent), "Please initialize the model before sampling."

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

        return solution

    @pyro.nn.pyro_method
    def add_observation_likelihoods(self, solution: Solution, observation_model=None) -> None:
        '''
        Compute likelihoods for observations.
        '''
        if observation_model is None:
            observation_model = self.observation_model

        for var_name in self._observation_var_names:
            observation_indices = self._observation_indices[var_name]
            observation_values = self._observation_values[var_name]
            filtered_solution = {v: solution[observation_indices] for v, solution in solution.items()}
            with pyro.condition(data={var_name: observation_values}):
                observation_model(filtered_solution, var_name)

    def log_solution(self, solution: Solution) -> Solution:
        '''
        This method wraps the solution in a pyro.deterministic call to ensure it is in the trace.
        '''
        # Log the solution
        logging_indices = [i for i, event in enumerate(self._static_events) if isinstance(event, LoggingEvent)]

        # Return the logged solution wrapped in a pyro.deterministic call to ensure it is in the trace
        logged_solution = {v: pyro.deterministic(f"{v}_sol", solution[logging_indices]) for v, solution in solution.items()}

        return logged_solution

## why is this here? It should be in MiraPetriNetODESystem if it is Mira specific
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
    return str(param.key)


class MiraPetriNetODESystem(PetriNetODESystem):
    """
    Create an ODE system from a petri-net specification.
    """
    def __init__(self, G: mira.modeling.Model):
        self.G = G
        super().__init__()

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
        return cls.from_askenet(model)

    @from_mira.register(mira.metamodel.TemplateModel)
    @classmethod
    def _from_template_model(cls, model_template: mira.metamodel.TemplateModel):
        return cls.from_askenet(model_template)

    @from_mira.register(dict)
    @classmethod
    def _from_json(cls, model_json: dict):
        return cls.from_mira(mira.metamodel.TemplateModel.from_json(model_json))

    @from_mira.register(str)
    @classmethod
    def _from_json_file(cls, model_json_path: str):
        if not os.path.exists(model_json_path):
            raise ValueError(f"Model file not found: {model_json_path}")
        with open(model_json_path, "r") as f:
            return cls.from_mira(json.load(f))

    @functools.singledispatchmethod
    @classmethod
    def from_askenet(cls, model: mira.modeling.Model) -> "MiraPetriNetODESystem":
        """Return a model from a MIRA model."""
        return cls(model)


    @from_askenet.register(mira.metamodel.TemplateModel)
    @classmethod
    def _from_template_model(cls, model_template: mira.metamodel.TemplateModel):
        """Return a model from a MIRA model template."""
        model = cls.from_askenet(mira.modeling.Model(model_template))

        # Check if all parameter names are strings
        if all(isinstance(param.key, str) for param in model.G.parameters.values()):
            return model
        else:
            new_template = aggregate_parameters(model_template)
            return cls.from_askenet(mira.modeling.Model(new_template))


    @from_askenet.register(dict)
    @classmethod
    def _from_json(cls, model_json: dict):
        """Return a model from an ASKEM Model Representation json."""
        if "templates" in model_json:
            return cls.from_askenet(mira.metamodel.TemplateModel.from_json(model_json))
        elif 'petrinet' in model_json['schema']:
            return cls.from_askenet(petrinet.template_model_from_askenet_json(model_json))
        elif 'regnet' in model_json['schema']:
            return cls.from_askenet(regnet.template_model_from_askenet_json(model_json))


    
    @from_askenet.register(str)
    @classmethod
    def _from_path(cls, model_json_path: str):
        """Return a model from an ASKEM Model Representation path (either url or local file)."""
        if "https://" in model_json_path:
            res = requests.get(model_json_path)
            model_json = res.json()
        else:
            if not os.path.exists(model_json_path):
                raise ValueError(f"Model file not found: {model_json_path}")
            with open(model_json_path) as fh:
                model_json = json.load(fh)
        return cls.from_askenet(model_json)

    def to_askenet_petrinet(self) -> dict:
        """Return an ASKEM Petrinet Model Representation json."""
        return AskeNetPetriNetModel(self.G).to_json()

    def to_askenet_regnet(self) -> dict:
        """Return an ASKEM Regnet Model Representation json."""
        return AskeNetRegNetModel(self.G).to_json()
    
    def to_networkx(self) -> networkx.MultiDiGraph:
        from pyciemss.utils.petri_utils import load
        return load(mira.modeling.petri.PetriNetModel(self.G).to_json())

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
                flux = flux * functools.reduce(operator.add, [states[k] for k in transition.control]) / population_size**len(transition.control)

            for c in transition.consumed:
                derivs[c] -= flux
            for p in transition.produced:
                derivs[p] += flux

        return tuple(derivs[v] for v in self.var_order.values())

    @pyro.nn.pyro_method
    def param_prior(self):
        for param_info in self.G.parameters.values():
            param_name = get_name(param_info)

            param_value = param_info.value
            if param_value is None:  # TODO remove this placeholder when MIRA is updated
                param_value = torch.nn.Parameter(torch.tensor(0.1))
            if isinstance(param_value, torch.nn.Parameter):
                setattr(self, param_name, pyro.param(param_name, param_value))
            elif isinstance(param_value, pyro.distributions.Distribution):
                # This used to be a pyro.nn.PyroSample, but that sampled repeatedly on each call to getattr.
                setattr(self, param_name, pyro.sample(param_name, param_value))
            elif isinstance(param_value, (int, float, numpy.ndarray, torch.Tensor)):
                self.register_buffer(param_name, torch.as_tensor(param_value))
            else:
                raise TypeError(f"Unknown parameter type: {type(param_value)}")

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
            elif param_value <= 0:
                warnings_string = f"Parameter {get_name(param_info)} has value {param_value} <= 0.0 and will be set to Uniform(0, 0.1)"
                warnings.warn(warnings_string)
                param_info.value = pyro.distributions.Uniform(0.0, 0.1)
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
        pyro.sample(var_name, ScaledBeta(mean, self.total_population, pseudocount).to_event(1))
