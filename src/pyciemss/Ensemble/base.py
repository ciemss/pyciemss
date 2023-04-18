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

        # Check that all models are of the same type.
        model_types = set([type(model) for model in self.models])
        assert(len(model_types) == 1)

    def reset(self):
        for model in self.models:
            model.reset()
    
    def forward(self):
        model_assignment = pyro.sample('model_assignment', pyro.distributions.Categorical(self.weights))
        return self.models[model_assignment].forward()
    

        
    