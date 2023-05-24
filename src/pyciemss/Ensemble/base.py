from typing import Callable, Dict, Sequence, Optional, OrderedDict, Union

import pyro
import torch
from pyro.contrib.autoname import name_count, scope
from torch import Tensor

from pyciemss.interfaces import DynamicalSystem

# TODO: refactor this to use a more general event class
from pyciemss.PetriNetODE.events import Event
from pyciemss.utils.distributions import ScaledBeta


class EnsembleSystem(DynamicalSystem):
    """
    Base class for ensembles of dynamical systems.
    """

    # TODO: add type hints for solution_mappings. It should be a mapping from dict to dict.
    def __init__(
        self,
        models: Sequence[DynamicalSystem],
        dirichlet_alpha: Tensor,
        solution_mappings: Sequence[Callable],
    ) -> None:
        self.models = models
        self.dirichlet_alpha = dirichlet_alpha
        self.solution_mappings = solution_mappings

        assert len(self.models) == self.dirichlet_alpha.shape[0]
        assert len(self.models) == len(self.solution_mappings)

        # Check that all models are of the same class.
        model_types = set([model.__class__ for model in self.models])
        assert len(model_types) == 1

        # TODO: Check that all of the solution mappings map to the same set of variables.
        super().__init__()

        # Set the method for adding observation likelihoods to be the same as one of the models in the ensemble.
        # Note: that self.models[j].__class__ is constant for all j, therefore we can use self.models[0].__class__ arbitrarily.
        # TODO: This will need a test.

    def add_observation_likelihoods(self, solution):
        return self.models[0].add_observation_likelihoods(
            solution, self.observation_model
        )

    def reset(self) -> None:
        # It's unclear why we need this given __getattr__ below, but it seems necessary...
        for model in self.models:
            model.reset()

    # TODO: this approach is extremely flexible, but not as discoverable as it could be.
    # This allows us to call a method on the ensemble and have it call the same method on all of the models in the ensemble.
    def __getattr__(self, attr: str):
        """
        Call the attribute or method of all of the models in the ensemble.
        """
        # Check if the attribute is callable.
        if callable(getattr(self.models[0], attr)):
            # If so, return a function that calls the attribute or method of all of the models in the ensemble.
            def call(*args, **kwargs):
                result = [
                    getattr(model, attr)(*args, **kwargs) for model in self.models
                ]
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
        """
        Prior distribution over model parameters.
        This avoids name collisions if multiple models have the same parameter name.
        """
        for i, model in enumerate(self.models):
            with scope(prefix=f"model_{i}"):
                model.param_prior()

    def log_solution(self, solution):
        """
        Log the solution of the ensemble.
        """
        # When we initialized the model we checked that all of the models are of the same class.
        # Therefore, we can call the log_solution method of any arbitrary model in the ensemble.
        return self.models[0].log_solution(solution)

    def get_solution(self, *args, **kwargs):
        """
        Get the solution of the ensemble.
        """
        model_weights = pyro.sample(
            "model_weights", pyro.distributions.Dirichlet(self.dirichlet_alpha)
        )

        solutions = [
            mapping(model.get_solution(*args, **kwargs))
            for model, mapping in zip(self.models, self.solution_mappings)
        ]

        # TODO: make this broadcasted instead of looping over the keys.
        solution = {
            k: sum([model_weights[i] * v[k] for i, v in enumerate(solutions)])
            for k in solutions[0].keys()
        }

        return solution

    def __repr__(self) -> str:
        return f"Ensemble of {len(self.models)} models. \n\n \tDirichlet Alpha: {self.dirichlet_alpha}. \n\n \tModels: {self.models}"


class ScaledBetaNoiseEnsembleSystem(EnsembleSystem):
    """
    Ensemble of dynamical systems with scaled beta noise.
    """

    def __init__(
        self,
        models: Sequence[DynamicalSystem],
        dirichlet_alpha: Tensor,
        solution_mappings: Sequence[Callable],
        total_population: float,
        pseudocount: float = 1.0,
    ) -> None:
        super().__init__(models, dirichlet_alpha, solution_mappings)
        self.total_population = total_population
        self.pseudocount = pseudocount

    @pyro.nn.pyro_method
    def observation_model(self, solution, var_name: str):
        """
        Observation model for the ensemble.
        """
        mean = solution[var_name]
        pyro.sample(
            var_name,
            ScaledBeta(mean, self.total_population, self.pseudocount).to_event(1),
        )

    def __rep__(self) -> str:
        return f"Scaled Beta Noise Ensemble of {len(self.models)} models. \n\n \tDirichlet Alpha: {self.dirichlet_alpha}. \n\n \tModels: {self.models} \n\n \tPseudocount: {self.pseudocount}"
