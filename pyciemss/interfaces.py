import functools
from typing import Generic, Optional, TypeVar, Union, Dict, Callable
from compiled_dynamics import CompiledDynamics

import pyro
import torch
import contextlib

from chirho.dynamical.ops import State
from chirho.dynamical.handlers import InterruptionEventLoop, LogTrajectory, StaticIntervention, DynamicIntervention


# By convention we use "T" to denote the type of the dynamical system, e.g. `ODE`, `PDE`, or `SDE`.
T = TypeVar("T")

# Type alias for the variational approximate (i.e. "guide") representing the approximate posterior distribution over parameters.
InferredParameters = pyro.nn.PyroModule



def load_model(model_path_or_json: Union[str, Dict]) -> CompiledDynamics:
    """
    Load a model from a path or a JSON string.
    """
    return CompiledDynamics.load(model_path_or_json)

def save_model(model: CompiledDynamics, model_path: str) -> None:
    """
    Save a model to a path.
    """
    model.save(model_path)

def simulate(
    model: CompiledDynamics,
    start_time: float,
    end_time: float,
    logging_step_size: float,
    inferred_parameters: Optional[InferredParameters] = None,
    static_interventions: Dict[float, Dict[str, torch.Tensor]] = {},
    dynamic_interventions: Dict[Callable[[State[torch.Tensor]], torch.Tensor], Dict[str, torch.Tensor]] = {},
) -> State[torch.Tensor]:
    """
    Simulate trajectories from a given `model`, conditional on specified `inferred_parameters` distribution.
    If `inferred_parameters` is not given, this will sample from the prior distribution.
    """

    timespan = torch.arange(start_time, end_time, logging_step_size)

    with LogTrajectory(timespan) as lt:
        with InterruptionEventLoop():
            # TODO: check this. Don't think it's correct usage of ExitStack
            with contextlib.ExitStack() as stack:
                for time, intervened_state_dict in static_interventions.items():
                    static_intervened_state = State(**intervened_state_dict)
                    stack.enter_context(StaticIntervention(torch.as_tensor(time), intervened_state))
                    for event_fn, intervened_state_dict in dynamic_interventions.items():
                        intervened_state = State(**intervened_state_dict)
                        stack.enter_context(DynamicIntervention(event_fn, intervened_state))
                model(start_time, end_time)

    return lt


@functools.singledispatch
def intervene(
    model: CompiledDynamics, intervention: Intervention[T], *args, **kwargs
) -> CompiledDynamics:
    """
    `intervene(model, intervention)` returns a new model where the intervention has been applied.
    """
    raise NotImplementedError


@functools.singledispatch
def calibrate(
    model: DynamicalSystem[T], data: Data[T], *args, **kwargs
) -> InferredParameters[T]:
    """
    Infer parameters for a DynamicalSystem model conditional on data.
    This is typically done using a variational approximation.
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
