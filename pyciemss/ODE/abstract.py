from typing import Iterable, Union, Dict, Tuple, TypeVar, Optional

import torch
import pyro
import torchdiffeq

from torch import nn
from pyro.nn import PyroModule, pyro_method

from torchdiffeq import odeint

T = TypeVar('T', torch.tensor, float)
S = TypeVar('S', torch.tensor, float)

class ODE(PyroModule):
    '''
    Base class for ordinary differential equations models in PyCIEMSS.
    '''

    def __init__(self):
        super().__init__()

    def deriv(self, t: T, state: S) -> S:
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
    def observation_model(self, solution: S, data: Optional[Dict[str, S]] = None) -> S:
        '''
        Conditional distribution of observations given true state trajectory.
        All random variables must be defined using `pyro.sample` or `PyroSample` methods.
        '''
        raise NotImplementedError

    @pyro_method
    def forward(self, initial_state: S, tspan: torch.tensor, data: Optional[Dict[str, S]] = None) -> Tuple[S, S]:
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