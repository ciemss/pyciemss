from typing import Iterable, Union, Dict

import torch
import torchdiffeq

from torch import nn
from pyro.nn import PyroModule


class ODEModel(PyroModule):
    def __init__(self, t0=0.0, adjoint=False):
        super().__init__()
        self.t0 = nn.Parameter(torch.as_tensor(t0))
        self._solver = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    
    def solve(self, initial_state, tspan):
        '''
        TODO: add docstring
        Note: `self` is executable.
        '''
        return self._solver(self, initial_state, tspan)

    def prior_sample(self):

        '''
        TODO: add docstring
        '''

        raise NotImplementedError

    def forward(self, t: torch.Tensor, state: Iterable[torch.Tensor]) -> Iterable[torch.Tensor]:
        '''
        `forward` defines the system of first order ordinary differential equations.
        Given an iterable of tensors `state` and a tensor `t`, `forward` returns an iterable of tensors `dstate`
        where each element is the resulting time derivative at `t` of the corresponding element in `state`.
        '''
        raise NotImplementedError


class ObservationModel(PyroModule):
    def __init__(self):
        super().__init__()

    def forward(self, solution: torch.Tensor, data: Dict[str, Union[torch.Tensor, None]]) -> torch.Tensor:
        '''
        TODO: add docstring
        '''
        raise NotImplementedError


class ODESolution(PyroModule):
    def __init__(self, 
                ode_model:ODEModel, 
                observation_model:ObservationModel
                ):
        super().__init__()
        self.ode_model = ode_model
        self.observation_model = observation_model

    def forward(self, initial_state, tspan, data=None):
        '''
        TODO: add docstring
        '''
        self.ode_model.prior_sample()
        solution = self.ode_model.solve(initial_state, tspan)
        observations = self.observation_model(solution, data)
        return observations