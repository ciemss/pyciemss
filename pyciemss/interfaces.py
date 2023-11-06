import contextlib
from typing import Any, Callable, Dict, Optional, Union

import pyro
import torch
from chirho.dynamical.handlers import (
    DynamicIntervention,
    InterruptionEventLoop,
    LogTrajectory,
    StaticIntervention,
)
from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.ops import State

from pyciemss.compiled_dynamics import CompiledDynamics
from pyciemss.integration_utils.custom_decorators import pyciemss_logging_wrapper
from pyciemss.integration_utils.result_processing import prepare_interchange_dictionary


@pyciemss_logging_wrapper
def sample(
    model_path_or_json: Union[str, Dict],
    end_time: float,
    logging_step_size: float,
    num_samples: int,
    *,
    solver_method="dopri5",
    solver_options: Dict[str, Any] = {},
    start_time: float = 0.0,
    inferred_parameters: Optional[pyro.nn.PyroModule] = None,
    static_interventions: Dict[float, Dict[str, torch.Tensor]] = {},
    dynamic_interventions: Dict[
        Callable[[Dict[str, torch.Tensor]], torch.Tensor], Dict[str, torch.Tensor]
    ] = {},
) -> Dict[str, torch.Tensor]:
    """
    Load a model from a file, compile it into a probabilistic program, and sample from it.

    Args:
        model_path_or_json: Union[str, Dict]
            - A path to a AMR model file or JSON containing a model in AMR form.
        end_time: float
            - The end time of the sampled simulation.
        logging_step_size: float
            - The step size to use for logging the trajectory.
        num_samples: int
            - The number of samples to draw from the model.
        solver_method: str
            - The method to use for solving the ODE. See torchdiffeq's `odeint` method for more details.
            - If performance is incredibly slow, we suggest using `euler` to debug.
              If using `euler` results in faster simulation, the issue is likely that the model is stiff.
        solver_options: Dict[str, Any]
            - Options to pass to the solver. See torchdiffeq' `odeint` method for more details.
        start_time: float
            - The start time of the model. This is used to align the `start_state` from the
              AMR model with the simulation timepoints.
            - By default we set the `start_time` to be 0.
        inferred_parameters: Optional[pyro.nn.PyroModule]
            - A Pyro module that contains the inferred parameters of the model.
              This is typically the result of `calibrate`.
            - If not provided, we will use the default values from the AMR model.
        static_interventions: Dict[float, Dict[str, torch.Tensor]]
            - A dictionary of static interventions to apply to the model.
            - Each key is the time at which the intervention is applied.
            - Each value is a dictionary of the form {state_variable_name: value}.
        dynamic_interventions: Dict[Callable[[Dict[str, torch.Tensor]], torch.Tensor], Dict[str, torch.Tensor]]
            - A dictionary of dynamic interventions to apply to the model.
            - Each key is a function that takes in the current state of the model and returns a tensor.
              When this function crosses 0, the dynamic intervention is applied.
            - Each value is a dictionary of the form {state_variable_name: value}.

    Returns:
        result: Dict[str, torch.Tensor]
            - Dictionary of outputs from the model.
                - Each key is the name of a parameter or state variable in the model.
                - Each value is a tensor of shape (num_samples, num_timepoints) for state variables
                    and (num_samples,) for parameters.
    """

    model = CompiledDynamics.load(model_path_or_json)

    timespan = torch.arange(start_time, end_time, logging_step_size)

    static_intervention_handlers = [
        StaticIntervention(time, State(**static_intervention_assignment))
        for time, static_intervention_assignment in static_interventions.items()
    ]
    dynamic_intervention_handlers = [
        DynamicIntervention(event_fn, State(**dynamic_intervention_assignment))
        for event_fn, dynamic_intervention_assignment in dynamic_interventions.items()
    ]

    def wrapped_model():
        with LogTrajectory(timespan) as lt:
            with InterruptionEventLoop():
                with contextlib.ExitStack() as stack:
                    for handler in (
                        static_intervention_handlers + dynamic_intervention_handlers
                    ):
                        stack.enter_context(handler)
                    model(
                        torch.as_tensor(start_time),
                        torch.tensor(end_time),
                        TorchDiffEq(method=solver_method, options=solver_options),
                    )
        # Adding deterministic nodes to the model so that we can access the trajectory in the Predictive object.
        [pyro.deterministic(f"{k}_state", v) for k, v in lt.trajectory.items()]

    samples = pyro.infer.Predictive(
        wrapped_model, guide=inferred_parameters, num_samples=num_samples
    )()

    return prepare_interchange_dictionary(samples)


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
