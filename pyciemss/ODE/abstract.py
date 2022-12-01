from typing import Iterable, Union, Dict

import torch
import pyro
import torchdiffeq

from torch import nn
from pyro.nn import PyroModule, pyro_method

from torchdiffeq import odeint


class ODE(PyroModule):
    def __init__(self):
        super().__init__()

    def deriv(self, t, state):
        '''
        TODO: add a docstring
        '''
        raise NotImplementedError

    @pyro_method
    def param_prior(self):
        '''
        TODO: add a docstring
        '''
        raise NotImplementedError

    @pyro_method
    def observation_model(self, solution, data):
        '''
        TODO: add a docstring
        '''
        raise NotImplementedError

    @pyro_method
    def forward(self, initial_state, tspan, data=None):
        
        # Sample parameters from the prior
        self.param_prior()

        # Simulate from ODE
        solution = odeint(self.deriv, initial_state, tspan)
        
        # Add Observation noise
        observations = self.observation_model(solution, data)

        return observations