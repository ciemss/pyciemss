import functools
import json
import operator
import os
from typing import Dict, List, Tuple, Type, TypeVar, Optional, Tuple, Union

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

from torchdiffeq import odeint

T = TypeVar('T')
Time = TypeVar('Time')
State = TypeVar('State')
Solution = TypeVar('Solution')
Observation = Solution

class ODE(pyro.nn.PyroModule):
    '''
    Base class for ordinary differential equations models in PyCIEMSS.
    '''
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
    def observation_model(self, solution: Solution, data: Optional[Dict[str, State]] = None) -> Observation:
        '''
        Conditional distribution of observations given true state trajectory.
        All random variables must be defined using `pyro.sample` or `PyroSample` methods.
        '''
        raise NotImplementedError

    def forward(self, initial_state: State, tspan: torch.tensor, data: Optional[Dict[str, Solution]] = None) -> Tuple[Solution, Observation]:
        '''
        Joint distribution over model parameters, trajectories, and noisy observations.
        '''

        # Sample parameters from the prior
        self.param_prior()

        # Simulate from ODE.
        # Constant deltaT method like `euler` necessary to get interventions without name collision.
        solution = odeint(self.deriv, initial_state, tspan, method="euler")

        # Add Observation noise
        observations = self.observation_model(solution, data)

        return solution, observations


@functools.singledispatch
def get_name(obj) -> str:
    """
    Function to get a string-valued name for a MIRA object for use in a Pyro model.
    """
    raise NotImplementedError


@get_name.register
def _get_name_str(name: str) -> str:
    return name


@get_name.register
def _get_name_variable(var: mira.modeling.Variable) -> str:
    return f"{var.key[0]}"


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
        self.var_order = tuple(sorted(G.variables.values(), key=get_name))
        self.named_variables = {get_name(var): var for var in self.var_order}

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

        if all(var.data.get("initial_value", None) is not None for var in self.var_order):
            for var in self.var_order:
                self.register_buffer(
                    f"default_initial_state_{get_name(var)}",
                    torch.as_tensor(var.data["initial_value"])
                )

            self.default_initial_state = tuple(
                getattr(self, f"default_initial_state_{get_name(var)}", None)
                for var in self.var_order
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
        states = {k: state[i] for i, k in enumerate(self.var_order)}
        derivs = {k: 0. for k in states}

        population_size = sum(states.values())

        for transition in self.G.transitions.values():
            flux = getattr(self, get_name(transition.rate)) * functools.reduce(
                operator.mul, [states[k] for k in transition.consumed], 1
            )
            if len(transition.control) > 0:
                flux = flux * sum([states[k] for k in transition.control]) / population_size

            flux = pyro.deterministic(f"flux_{get_name(transition)} {t}", flux, event_dim=0)

            for c in transition.consumed:
                derivs[c] -= flux
            for p in transition.produced:
                derivs[p] += flux

        return tuple(pyro.deterministic(f"ddt_{get_name(v)} {t}", derivs[v], event_dim=0) for v in self.var_order)

    @pyro.nn.pyro_method
    def observation_model(self, solution: Solution, data: Optional[Dict[str, State]] = None) -> Observation:
        with pyro.condition(data=data if data is not None else {}):
            return tuple(
                pyro.deterministic(f"obs_{get_name(var)}", sol, event_dim=1)
                for var, sol in zip(self.var_order, solution)
            )
