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


def sample(
    model_path_or_json: Union[str, Dict],
    start_time: float,
    end_time: float,
    logging_step_size: float,
    num_samples: int,
    *,
    inferred_parameters: Optional[pyro.nn.PyroModule] = None,
    static_interventions: Dict[float, Dict[str, torch.Tensor]] = {},
    dynamic_interventions: Dict[
        Callable[[State[torch.Tensor]], torch.Tensor], Dict[str, torch.Tensor]
    ] = {},
) -> State[torch.Tensor]:
    """
    Load a model from a file, compile it into a probabilistic program, and sample from it.

    Args:
        petri_model_or_path: Union[str, mira.metamodel.TemplateModel, mira.modeling.Model]
            - A path to a petri net file, or a petri net object.
            - This path can be a URL or a local path to a mira model or AMR model.
            - Alternatively, this can be a mira template model directly.
        num_samples: int
            - The number of samples to draw from the model.
        timepoints: [Iterable[float]]
            - The timepoints to simulate the model from.
              Backcasting and/or forecasting is reflected in the choice of timepoints.
        interventions: Optional[Iterable[Tuple[float, str, float]]]
            - A list of interventions to apply to the model.
              Each intervention is a tuple of the form (time, parameter_name, value).
        start_state: Optional[dict[str, float]]
            - The initial state of the model. If None, the initial state is taken from the mira model.
        start_time: float
            - The start time of the model. This is used to align the `start_state` with the `timepoints`.
            - By default we set the `start_time` to be a small negative number to avoid numerical
              issues w/ collision with the `timepoints` which typically start at 0.
        method: str
            - The method to use for solving the ODE. See torchdiffeq's `odeint` method for more details.
            - If performance is incredibly slow, we suggest using `euler` to debug.
              If using `euler` results in faster simulation, the issue is likely that the model is stiff.
        compile_rate_law_p: bool
            - Whether or not to compile the rate law from the AMR rate expression. default True.
              False is useful for debugging.
        compile_observables_p: bool
            - Whether or not to compile the observables from the AMR observable expression. default True.
              False is useful for debugging.
        time_unit: str
            - Time unit (used for labeling outputs)
        visual_options: None, bool, dict[str, any]
            - True output a visual
            - False do not output a visual
            - dict output a visual with the dictionary passed to the visualization as kwargs
        alpha_qs: Optional[Iterable[float]]
            - The quantiles required for estimating weighted interval score to test forecasting accuracy.
        stacking_order: Optional[str]
            - The stacking order requested for the quantiles to keep the selected quantity together for each state.
            - Options: "timepoints" or "quantiles"

    Returns:
        result: dict
            - Dictionary of outputs with following attribute:
                * data: The samples from the model as a pandas DataFrame.
                * quantiles: The quantiles for ensemble score calculation as a pandas DataFrames.
                * risk: Risk estimates for each state as 2-day average at the final timepoint
                    * risk: Estimated alpha-superquantile risk with alpha=0.95
                    * qoi: Samples of quantity of interest
                          (in this case, 2-day average of the state at the final timepoint)
                * visual: Visualization. (If visual_options is truthy)
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
                    model(torch.as_tensor(start_time), torch.tensor(end_time))
        # Adding deterministic nodes to the model so that we can access the trajectory in the Predictive object.
        [pyro.deterministic(f"state_{k}", v) for k, v in lt.trajectory.items()]

    return pyro.infer.Predictive(
        wrapped_model, guide=inferred_parameters, num_samples=num_samples
    )()


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
