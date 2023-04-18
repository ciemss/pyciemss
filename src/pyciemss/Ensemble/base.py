from pyciemss import DynamicalSystem

import pyro

from typing import Dict, List, Optional, Union, OrderedDict

class Ensemble(DynamicalSystem):
    '''
    Base class for ensembles of dynamical systems.
    '''

    def __init__(self, models: List[DynamicalSystem], weights: List[float]) -> None:
        super().__init__()
        self.models = models
        self.weights = weights
        assert(len(self.models) == len(self.weights))

    def reset(self):
        for model in self.models:
            model.reset()
    
    def forward(self):
        model_assignment = pyro.sample('model_assignment', pyro.distributions.Categorical(self.weights))
        return self.models[model_assignment].forward()
    

        
    