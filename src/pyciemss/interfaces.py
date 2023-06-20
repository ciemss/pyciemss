import functools
from typing import Generic, Optional, TypeVar

import pyro

T = TypeVar("T")


# TODO: Figure out how to declare the parameteric type of `DynamicalSystem` in the signature.
class DynamicalSystem(pyro.nn.PyroModule):
    """
    Base class for dynamical systems.
    """

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self):
        raise NotImplementedError

    def setup_before_solve(self):
        raise NotImplementedError

    def param_prior(self):
        raise NotImplementedError

    def get_solution(self, *args, **kwargs):
        raise NotImplementedError

    def add_observation_likelihoods(self, solution: T):
        raise NotImplementedError

    def log_solution(self, solution: T) -> T:
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """
        Joint distribution over model parameters, trajectories, and noisy observations.
        """
        # Setup the anything the dynamical system needs before solving.
        self.setup_before_solve()

        # Sample parameters from the prior
        self.param_prior()

        # Solve the ODE
        solution = self.get_solution(*args, **kwargs)

        # Add the observation likelihoods
        self.add_observation_likelihoods(solution)

        return self.log_solution(solution)


# TODO: Figure out how to declare the parameteric type of `DynamicalSystem` in the signature.
@functools.singledispatch
def setup_model(model: DynamicalSystem, *args, **kwargs) -> DynamicalSystem:
    """
    Instatiate a model for a particular configuration of initial conditions, boundary conditions, logging events, etc.
    """
    raise NotImplementedError


# TODO: Figure out how to declare the parameteric type of `DynamicalSystem` in the signature.
@functools.singledispatch
def reset_model(model: DynamicalSystem, *args, **kwargs) -> DynamicalSystem:
    """
    Reset a model to its initial state.
    reset_model * setup_model = id
    """
    raise NotImplementedError


# TODO: Figure out how to declare the parameteric type of `DynamicalSystem` in the signature.
@functools.singledispatch
def intervene(model: DynamicalSystem, intervention, *args, **kwargs) -> DynamicalSystem:
    """
    `intervene(model, intervention)` returns a new model where the intervention has been applied.
    """
    raise NotImplementedError


@functools.singledispatch
def assert_observations_valid(model: DynamicalSystem, data, *args, **kwargs) -> None:
    """
    Check that the observations are valid for the given model.
    """
    raise NotImplementedError


# TODO: Figure out how to declare the parameteric type of `DynamicalSystem` in the signature.
@functools.singledispatch
def calibrate(model: DynamicalSystem, data, *args, **kwargs):
    """
    Infer parameters for a DynamicalSystem model conditional on data. This is typically done using a variational approximation.
    """
    raise NotImplementedError


# TODO: Figure out how to declare the parameteric type of `DynamicalSystem` in the signature.
@functools.singledispatch
def sample(
    model: DynamicalSystem, inferred_parameters: Optional[T] = None, *args, **kwargs
):
    """
    Sample trajectories from a given `model`, conditional on specified `inferred_parameters`. If `inferred_parameters` is not given, this will sample from the prior distribution.
    """
    raise NotImplementedError


# TODO: Figure out how to declare the parameteric type of `DynamicalSystem` in the signature.
@functools.singledispatch
def optimize(
    model: DynamicalSystem,
    objective_function,
    constraints,
    optimization_algorithm,
    *args,
    **kwargs
):
    """
    Optimize the objective function subject to the constraints.
    """
    raise NotImplementedError
