import collections
import functools
import json
import operator
import os
from typing import Dict, List, Tuple, Type, TypeVar, Optional, Tuple, Union, OrderedDict, Callable

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

import heapq

import torch

from torchdiffeq import odeint

from pyciemss.ODE.events import StaticEvent, StartEvent, ObservationEvent, LoggingEvent, DynamicStopEvent

T = TypeVar('T')
Time = TypeVar('Time')
State = TypeVar('State')
Solution = TypeVar('Solution')
Observation = Solution

class ODE(pyro.nn.PyroModule):
    '''
    Base class for ordinary differential equations models in PyCIEMSS.
    '''

    def __init__(self):
        super().__init__()

        self._start_event = None
        self._observation_events = []
        self._logging_events = []
        
        self._static_events = []
        self._dynamic_stop_events = []

    @functools.singledispatchmethod
    def load_start_event(self, start_event: StartEvent) -> None:
        '''
        Loads a start event into the model.
        '''
        self._start_event = start_event
        self._construct_static_events()

    @load_start_event.register
    def _load_start_event_float(self, time: float, state: Dict[str, Union[str, torch.Tensor]]) -> None:
        state = {k: torch.tensor(v) for k, v in state.items()}
        self.load_start_event(StartEvent(time, state))

    @load_start_event.register
    def _load_start_event_int(self, time: int, state: Dict[str, Union[str, torch.Tensor]]) -> None:
        state = {k: torch.tensor(v) for k, v in state.items()}
        self.load_start_event(StartEvent(float(time), state))

    def load_logging_events(self, times: Union[torch.Tensor, List]) -> None:
        '''
        Loads a list of logging events into the model.
        '''
        logging_events = [LoggingEvent(t) for t in times]
        self._logging_events = sorted(logging_events)
        self._construct_static_events()

    def delete_logging_events(self) -> None:
        '''
        Deletes all logging events from the model.
        '''
        self._logging_events = []
        self._construct_static_events()

    @functools.singledispatchmethod
    def load_observation_events(self, observation_events: list) -> None:
        # TODO: finish.
        # TODO: having trouble dispatching on list of ObservationEvents.
        '''
        Loads a list of observation events into the model.
        '''
        self._observation_events = sorted(observation_events)
        self._construct_static_events()

    def delete_observation_events(self) -> None:
        '''
        Deletes all observation events from the model.
        '''
        self._observation_events = []
        self._construct_static_events()

    def _construct_static_events(self):
        '''
        Returns a list of static events sorted by time.
        This will be called again when we condition the model on new observations.
        '''

        # Sort the static events by time. This is linear in the number of events. Assumes each are already sorted.
        self._static_events = [e for e in heapq.merge(*[[self._start_event], self._logging_events, self._observation_events])]

    def deriv(self, t: Time, state: State) -> State:
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
    def observation_model(self, solution: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor]) -> None:
        '''
        Conditional distribution of observations given true state trajectory.
        All random variables must be defined using `pyro.sample` or `PyroSample` methods.
        This needs to be called once for each `var_name` in the set of observed variables.
        '''
        raise NotImplementedError

    def var_order(self) -> OrderedDict[str, int]:
        '''
        Returns a dictionary mapping variable names to their order in the state vector.
        '''
        raise NotImplementedError

    def forward(self, method="dopri5", **kwargs) -> Dict[str, Solution]:
        '''
        Joint distribution over model parameters, trajectories, and noisy observations.
        '''
        # Sample parameters from the prior
        self.param_prior()

        # Check that the start event is the first event
        assert isinstance(self._static_events[0], StartEvent)

        # Sample initial state from the prior
        initial_state = tuple(self._static_events[0].initial_state[v] for v in self.var_order.keys())

        # Get tspan from static events
        tspan = torch.tensor([e.time for e in self._static_events])

        # Simulate from ODE
        solution = odeint(self.deriv, initial_state, tspan, method=method, **kwargs)
        solution = {v: solution[i] for i, v in enumerate(self.var_order.keys())}
    
        observation_var_names = set([event.var_name for event in self._static_events if isinstance(event, ObservationEvent)])

        # Compute likelihoods for observations
        for var_name in observation_var_names:
            observation_indices = [i for i, event in enumerate(self._static_events) if isinstance(event, ObservationEvent) and event.var_name == var_name]
            observation_values = torch.stack([self._static_events[i].observation for i in observation_indices])
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


class PetriNetODESystem(ODE):
    """
    Create an ODE system from a petri-net specification.
    """
    def __init__(self, G: mira.modeling.Model):
        super().__init__()
        self.G = G
        self.var_order = collections.OrderedDict(
            (get_name(var), var) for var in sorted(G.variables.values(), key=get_name)
        )

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

    @functools.singledispatchmethod
    @classmethod
    def from_mira(cls, model: mira.modeling.Model) -> "PetriNetODESystem":
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
    def deriv(self, t: T, state: Tuple[T, ...]) -> Tuple[T, ...]:
        states = {k: state[i] for i, k in enumerate(self.var_order.values())}
        derivs = {k: 0. for k in states}

        population_size = sum(states.values())

        for transition in self.G.transitions.values():
            flux = getattr(self, get_name(transition.rate)) * functools.reduce(
                operator.mul, [states[k] for k in transition.consumed], 1
            )
            if len(transition.control) > 0:
                flux = flux * sum([states[k] for k in transition.control]) / population_size

            # flux = pyro.deterministic(f"flux_{get_name(transition)} {t}", flux, event_dim=0)

            for c in transition.consumed:
                derivs[c] -= flux
            for p in transition.produced:
                derivs[p] += flux

        return tuple(derivs[v] for v in self.var_order.values())
        # return tuple(pyro.deterministic(f"ddt_{get_name(v)} {t}", derivs[v], event_dim=0) for v in self.var_order.values())

    # @pyro.nn.pyro_method
    # def observation_model(self, solution: Solution, data: Optional[Dict[str, State]] = None) -> Observation:
    #     with pyro.condition(data=data if data is not None else {}):
    #         return tuple(
    #             pyro.deterministic(f"obs_{get_name(var)}", sol, event_dim=1)
    #             for var, sol in zip(self.var_order.values(), solution)
    #         )

class GaussianNoisePetriNetODESystem(PetriNetODESystem):
    '''
    This is a wrapper around PetriNetODESystem that adds Gaussian noise to the ODE system.
    Additionally, this wrapper adds a uniform prior on the model parameters.
    '''
    def __init__(self, G: mira.modeling.Model, noise_var: float = 1):
        super().__init__(G)
        self.register_buffer("noise_var", torch.as_tensor(noise_var))

    @pyro.nn.pyro_method
    def param_prior(self):
        # Uniform priors on model parameters
        # lower bound = max(0.9 * value, 0)
        # upper bound = 1.1 * value

        for param_info in self.G.parameters.values():
            param_name = get_name(param_info)
            param_value = param_info.value
            if not isinstance(param_value, pyro.distributions.Distribution):
                val = pyro.sample(
                    param_name,
                    pyro.distributions.Uniform(max(0.9 * param_value, 0.0), 1.1 * param_value)
                )
                setattr(self, param_name, val)

    @pyro.nn.pyro_method
    def observation_model(self, solution: Dict[str, torch.Tensor], var_name: str) -> None:
        pyro.sample(var_name, pyro.distributions.Normal(solution[var_name], self.noise_var).to_event(1))