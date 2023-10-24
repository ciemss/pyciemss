import functools
from typing import Generic, Optional, TypeVar

import pyro

# By convention we use "T" to denote the type of the dynamical system, e.g. `ODE`, `PDE`, or `SDE`.
T = TypeVar("T")


class DynamicalSystem(Generic[T], pyro.nn.PyroModule):
    """
    A dynamical system is a model of a system that evolves over time.
    """

    pass


class Intervention(Generic[T]):
    """
    An intervention is a change to a dynamical system that is not a change to the parameters.
    """

    pass


class Data(Generic[T]):
    """
    Data is a collection of observations of the dynamical system.
    """

    pass


class InferredParameters(Generic[T], pyro.nn.PyroModule):
    """
    InferredParameters are the parameters of the dynamical system that are inferred from data.
    This will always be a pyro.nn.PyroModule, as we rely on Pyro's variational inference with AutoGuides.
    """

    pass


class Simulation(Generic[T]):
    """
    A simulation is a collection of trajectories of a dynamical system.
    """

    pass


class ObjectiveFunction(Generic[T]):
    """
    An objective function is a function that is optimized to infer parameters from data.
    """

    pass


class Constraints(Generic[T]):
    """
    Constraints are constraints on the parameters of the dynamical system for optimization.
    """

    pass


class OptimizationAlgorithm(Generic[T]):
    """
    An optimization algorithm is an algorithm that is used to optimize the objective function subject to constraints.
    """

    pass


class OptimizationResult(Generic[T]):
    """
    An optimization result is the result of optimizing the objective function subject to constraints.
    """

    pass


@functools.singledispatch
def setup_model(model: DynamicalSystem[T], *args, **kwargs) -> DynamicalSystem[T]:
    """
    Instatiate a model for a particular configuration of initial conditions, boundary conditions, logging events, etc.
    """
    raise NotImplementedError


@functools.singledispatch
def reset_model(model: DynamicalSystem[T], *args, **kwargs) -> DynamicalSystem[T]:
    """
    Reset a model to its initial state.
    reset_model * setup_model = id
    """
    raise NotImplementedError


@functools.singledispatch
def intervene(
    model: DynamicalSystem[T], intervention: Intervention[T], *args, **kwargs
) -> DynamicalSystem[T]:
    """
    `intervene(model, intervention)` returns a new model where the intervention has been applied.
    """
    raise NotImplementedError


@functools.singledispatch
def calibrate(
    model: DynamicalSystem[T], data: Data[T], *args, **kwargs
) -> InferredParameters[T]:
    """
    Infer parameters for a DynamicalSystem model conditional on data. This is typically done using a variational approximation.
    """
    raise NotImplementedError


@functools.singledispatch
def simulate(
    model: DynamicalSystem[T],
    inferred_parameters: Optional[InferredParameters[T]] = None,
    *args,
    **kwargs
) -> Simulation[T]:
    """
    Simulate trajectories from a given `model`, conditional on specified `inferred_parameters` distribution. If `inferred_parameters` is not given, this will sample from the prior distribution.
    """
    raise NotImplementedError


@functools.singledispatch
def optimize(
    model: DynamicalSystem[T],
    objective_function: ObjectiveFunction[T],
    constraints: Constraints[T],
    optimization_algorithm: OptimizationAlgorithm[T],
    *args,
    **kwargs
) -> OptimizationResult[T]:
    """
    Optimize the objective function subject to the constraints.
    """
    raise NotImplementedError
