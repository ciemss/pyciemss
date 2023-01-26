from typing import Dict, Tuple, TypeVar, Optional, Tuple

import torch
import pyro
import torchdiffeq

from torch import nn
from pyro.nn import PyroModule, pyro_method

from torchdiffeq import odeint

Time = TypeVar('Time', torch.tensor, float)
State = TypeVar('State', torch.tensor, float)
Solution = TypeVar('Solution', Tuple[State], State)
Observation = Solution

class ODE(PyroModule):
    '''
    Base class for ordinary differential equations models in PyCIEMSS.
    '''
    def __init__(self):
        super().__init__()

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

    @pyro_method
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
