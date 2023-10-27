import contextlib
from typing import Callable, Dict, Optional, Union

import pyro
import torch
from chirho.dynamical.handlers import (
    DynamicIntervention,
    InterruptionEventLoop,
    LogTrajectory,
    StaticIntervention,
)
from chirho.dynamical.ops import State

from pyciemss.compiled_dynamics import CompiledDynamics


def simulate(
    model_path_or_json: Union[str, Dict],
    start_time: float,
    end_time: float,
    logging_step_size: float,
    inferred_parameters: Optional[pyro.nn.PyroModule] = None,
    static_interventions: Dict[float, Dict[str, torch.Tensor]] = {},
    dynamic_interventions: Dict[
        Callable[[State[torch.Tensor]], torch.Tensor], Dict[str, torch.Tensor]
    ] = {},
) -> State[torch.Tensor]:
    """
    Simulate trajectories from a given `model`, conditional on specified `inferred_parameters` distribution.
    If `inferred_parameters` is not given, this will sample from the prior distribution.
    """

    model = CompiledDynamics.load(model_path_or_json)

    timespan = torch.arange(start_time, end_time, logging_step_size)

    with LogTrajectory(timespan) as lt:
        with InterruptionEventLoop():
            # TODO: check this. Don't think it's correct usage of ExitStack
            with contextlib.ExitStack() as stack:
                for time, static_intervened_state_dict in static_interventions.items():
                    static_intervened_state = State(**static_intervened_state_dict)
                    stack.enter_context(
                        StaticIntervention(
                            torch.as_tensor(time), static_intervened_state
                        )
                    )
                    for (
                        event_fn,
                        dynamic_intervened_state_dict,
                    ) in dynamic_interventions.items():
                        dynamic_intervened_state = State(
                            **dynamic_intervened_state_dict
                        )
                        stack.enter_context(
                            DynamicIntervention(event_fn, dynamic_intervened_state)
                        )
                        model(start_time, end_time)

    return lt.trajectory


# # TODO
# def calibrate(
#     model: CompiledDynamics, data: Data, *args, **kwargs
# ) -> pyro.nn.PyroModule:
#     """
#     Infer parameters for a DynamicalSystem model conditional on data.
#     This is typically done using a variational approximation.
#     """
#     raise NotImplementedError


# # TODO
# def optimize(
#     model: CompiledDynamics,
#     objective_function: ObjectiveFunction,
#     constraints: Constraints,
#     optimization_algorithm: OptimizationAlgorithm,
#     *args,
#     **kwargs
# ) -> OptimizationResult:
#     """
#     Optimize the objective function subject to the constraints.
#     """
#     raise NotImplementedError
