import contextlib
from typing import Callable, Dict, Optional, Union, Iterable

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
    num_samples: int,
    *,
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


def calibrate(
    model_path_or_json: Union[str, Dict],
    data_path: str,
    start_time: float,
    *,
    noise_model: str = "scaled_normal",
    noise_scale: float = 0.1,
    static_interventions: Dict[float, Dict[str, torch.Tensor]] = {},
    dynamic_interventions: Dict[
        Callable[[State[torch.Tensor]], torch.Tensor], Dict[str, torch.Tensor]
    ] = {},
    num_iterations: int = 1000,
    lr: float = 0.03,
    verbose: bool = False,
    num_particles: int = 1,
    deterministic_learnable_parameters: Iterable[str] = [],
    method="dopri5",
) -> pyro.nn.PyroModule:
    """
    Infer parameters for a DynamicalSystem model conditional on data.
    This uses variational inference with a mean-field variational family to infer the parameters of the model.
    """
        
    model = CompiledDynamics.load(model_path_or_json)

    def autoguide(model):
        guide = pyro.infer.autoguide.AutoGuideList(model)
        guide.append(
            pyro.infer.autoguide.AutoDelta(
                pyro.poutine.block(model, expose=deterministic_learnable_parameters)
            )
        )
        guide.append(
            pyro.infer.autoguide.AutoLowRankMultivariateNormal(
                pyro.poutine.block(model, hide=deterministic_learnable_parameters)
            )
        )
        return guide
    
    # TODO
    # end_time = ...

    static_intervention_handlers = [
        StaticIntervention(time, State(**static_intervention_assignment))
        for time, static_intervention_assignment in static_interventions.items()
    ]
    dynamic_intervention_handlers = [
        DynamicIntervention(event_fn, State(**dynamic_intervention_assignment))
        for event_fn, dynamic_intervention_assignment in dynamic_interventions.items()
    ]
    
    def wrapped_model():
        with InterruptionEventLoop():
            with contextlib.ExitStack() as stack:
                for handler in (
                    static_intervention_handlers + dynamic_intervention_handlers
                ):
                    stack.enter_context(handler)
                model(torch.as_tensor(start_time), torch.tensor(end_time))


    guide = autoguide(wrapped_model)
    optim = pyro.optim.Adam({"lr": lr})
    loss = pyro.infer.Trace_ELBO(num_particles=num_particles)
    svi = pyro.infer.SVI(wrapped_model, guide, optim, loss=loss)

    pyro.clear_param_store()

    for i in range(num_iterations):
        loss = svi.step(method=method)
        if verbose:
            if i % 25 == 0:
                print(f"iteration {i}: loss = {loss}")

    return guide


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
