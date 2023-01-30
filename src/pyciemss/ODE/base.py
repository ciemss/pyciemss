import functools
import numbers
import operator
from typing import Dict, Tuple, TypeVar, Optional, Tuple

import networkx
import numpy
import torch
import pyro

import mira
import mira.modeling
import mira.modeling.petri
import mira.metamodel

from torch import nn
from pyro.nn import PyroModule, pyro_method

from torchdiffeq import odeint

T = TypeVar('T')
Time = TypeVar('Time')
State = TypeVar('State')
Solution = TypeVar('Solution')
Observation = Solution

class ODE(PyroModule):
    '''
    Base class for ordinary differential equations models in PyCIEMSS.
    '''
    def deriv(self, t: Time, state: State) -> State:
        '''
        Returns a derivate of `state` with respect to `t`.
        '''
        raise NotImplementedError

    @pyro_method
    def param_prior(self) -> None:
        '''
        Inplace method defining the prior distribution over model parameters.
        All random variables must be defined using `pyro.sample` or `PyroSample` methods.
        '''
        raise NotImplementedError

    @pyro_method
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


class PetriNetODESystem(ODE):
    """Create an ODE system from a petri-net definition.

    Args:
        G: Petri-net graph

    Returns
        Callable: Function that takes a list of state values and returns a list of derivatives.
    """
    def __init__(self, Gm: mira.modeling.Model):
        super().__init__()
        self.Gm = Gm

        self.var_order = [v.key[0] for v in sorted(Gm.variables, key=lambda v: v.key)]

        for param_name, param_info in self.Gm.parameters.items():
            param_value = param_info.value
            if isinstance(param_value, torch.nn.Parameter):
                setattr(self, param_name, pyro.nn.PyroParam(param_value))
            elif isinstance(param_value, pyro.distributions.Distribution):
                setattr(self, param_name, pyro.nn.PyroSample(param_value))
            elif isinstance(param_value, (numbers.Number, numpy.ndarray, torch.Tensor)):
                self.register_buffer(param_name, torch.as_tensor(param_value))
            else:
                raise TypeError(f"Unknown parameter type: {type(param_value)}")

    @functools.singledispatchmethod
    @classmethod
    def from_model(cls, model: mira.modeling.Model) -> "PetriNetODESystem":
        return cls(model)

    @from_model.register
    @classmethod
    def _from_template_model(cls, model: mira.metamodel.TemplateModel):
        return cls(mira.modeling.Model(model))

    @from_model.register
    @classmethod
    def _from_json_string(cls, model: str):
        model = mira.sources.petri.template_model_from_petri_json(model)
        return cls(mira.modeling.Model(model))

    def to_networkx(self) -> networkx.MultiDiGraph:
        from pyciemss.utils.petri_utils import load
        return load(mira.modeling.petri.PetriNetModel(self.Gm).to_json())

    @pyro.nn.pyro_method
    def param_prior(self):
        for param_name in self.Gm.parameters.keys():
            getattr(self, param_name)

    @pyro.nn.pyro_method
    def deriv(self, t: T, state: Tuple[T, ...]) -> Tuple[T, ...]:
        states = {k: state[i] for i, k in enumerate(self.var_order)}
        derivs = {k: 0. for k in states}

        for transition in self.Gm.transitions.values():
            rate_param = getattr(self, transition.rate.key)
            flux = rate_param * functools.reduce(
                operator.mul, [states[k] for k in transition.consumed], 1
            ) * functools.reduce(
                operator.mul, [states[k] for k in transition.control], 1
            )
            flux = pyro.deterministic(f"{transition.name}_flux {t}", flux, event_dim=0)
            for c in transition.consumed:
                derivs[c.key[0]] -= flux
            for p in transition.produced:
                derivs[p.key[0]] += flux

        return tuple(pyro.deterministic(f"d{v}_dt_{t}", derivs[v], event_dim=0) for v in self.var_order)

    @pyro.nn.pyro_method
    def observation_model(self, solution: Solution, data: Optional[Dict[str, State]] = None) -> Solution:
        with pyro.condition(data=data if data is not None else {}):
            output = {}
            for name, value in zip(self.var_order, solution):
                output[name] = pyro.sample(
                    f"{name}_obs",
                    pyro.distributions.Normal(value, self.noise_var).to_event(1),
                )
            return output
