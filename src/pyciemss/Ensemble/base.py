from pyciemss.interfaces import DynamicalSystem

import pyro

from typing import Dict, List, Optional, Union, OrderedDict

# TODO: refactor this to use a more general event class
from pyciemss.PetriNetODE.events import Event

class EnsembleSystem(DynamicalSystem):
    '''
    Base class for ensembles of dynamical systems.
    '''

    def __init__(self, models: List[DynamicalSystem], weights: List[float]) -> None:
        self.models = models
        self.weights = weights
        
        assert(len(self.models) == len(self.weights))

        # Check that all models are of the same type.
        model_types = set([type(model) for model in self.models])
        assert(len(model_types) == 1)

        super().__init__()

    # TODO: this approach is extremely flexible, but not as discoverable as it could be.
    # This allows us to call a method on the ensemble and have it call the same method on all of the models in the ensemble.
    def __getattr__(self, attr: str):
        '''
        Call the attribute or method of all of the models in the ensemble.
        '''
        # Check if the attribute is callable.
        if callable(getattr(self.models[0], attr)):
            # If so, return a function that calls the attribute or method of all of the models in the ensemble.
            def call(*args, **kwargs):
                result = [getattr(model, attr)(*args, **kwargs) for model in self.models]
                # If all of the results are None, return None.
                if all([isinstance(r, type(None)) for r in result]):
                    return None
                else:
                    return result

            return call
        else:
            return [getattr(model, attr) for model in self.models]
    
    def forward(self):
        model_assignment = pyro.sample('model_assignment', pyro.distributions.Categorical(self.weights))
        return self.models[model_assignment].forward()
    
    def __repr__(self) -> str:
        return f'Ensemble of {len(self.models)} models. \n\n \tWeights: {self.weights}. \n\n \tModels: {self.models}'