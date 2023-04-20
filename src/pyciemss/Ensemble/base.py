from pyciemss.interfaces import DynamicalSystem

import pyro
import torch

from pyro.contrib.autoname import scope, name_count
from typing import Dict, List, Optional, Union, OrderedDict, Callable

# TODO: refactor this to use a more general event class
from pyciemss.PetriNetODE.events import Event

class EnsembleSystem(DynamicalSystem):
    '''
    Base class for ensembles of dynamical systems.
    '''
    # TODO: add type hints for solution_mappings. It should be a mapping from dict to dict.
    def __init__(self, 
                 models: List[DynamicalSystem], 
                 weights: torch.tensor,
                 solution_mappings: Callable) -> None:
        self.models = models
        self.weights = weights
        self.solution_mappings = solution_mappings
        
        assert(len(self.models) == len(self.weights) == len(self.solution_mappings))

        # Check that all models are of the same type.
        model_types = set([type(model) for model in self.models])
        assert(len(model_types) == 1)

        #TODO: Check that all of the solution mappings map to the same set of variables.
        super().__init__()

    def reset(self) -> None:
        # It's unclear why we need this given __getattr__ below, but it seems necessary...
        for model in self.models:
            model.reset()

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
    
    def setup_before_solve(self):
        for model in self.models:
            model.setup_before_solve()

    def param_prior(self):
        '''
        Prior distribution over model parameters.
        This avoids name collisions if multiple models have the same parameter name.
        '''
        for i, model in enumerate(self.models):
            with scope(prefix=f'model_{i}'):
                model.param_prior()
            
        
    def log_solution(self, solution):
        '''
        Log the solution of the ensemble.
        '''
        return solution
        
    def get_solution(self, *args, **kwargs):
        '''
        Get the solution of the ensemble.
        '''
        model_weights = pyro.sample('model_weights', pyro.distributions.Dirichlet(self.weights))

        solutions = [mapping(model.get_solution(*args, **kwargs)) for model, mapping in zip(self.models, self.solution_mappings)]

        solution = {k: sum([model_weights[i] * v[k] for i, v in enumerate(solutions)]) for k in solutions[0].keys()}

        return solution


    def add_observation_likelihoods(self, solution):
        # This will be a bit tricky, and we'll probably need to rethink how we map solutions to observations.
        pass
        
    
    def __repr__(self) -> str:
        return f'Ensemble of {len(self.models)} models. \n\n \tWeights: {self.weights}. \n\n \tModels: {self.models}'