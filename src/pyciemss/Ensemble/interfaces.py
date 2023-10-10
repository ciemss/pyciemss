import pyro
import torch

from pyro.infer import Predictive

from pyciemss.PetriNetODE.base import get_name
from pyciemss.interfaces import (
    setup_model,
    reset_model,
    intervene,
    sample,
    calibrate,
    DynamicalSystem,
    prepare_interchange_dictionary,
    DEFAULT_QUANTILES,
)

from pyciemss.utils import interface_utils

from pyciemss.Ensemble.base import (
    EnsembleSystem,
    ScaledBetaNoiseEnsembleSystem,
    ScaledNormalNoiseEnsembleSystem,
)

from typing import Optional, Tuple, Callable, Union
from collections.abc import Iterable
import copy

# TODO: probably refactor this out later.
from pyciemss.PetriNetODE.events import (
    StartEvent,
    ObservationEvent,
    LoggingEvent,
)

from pyciemss.custom_decorators import pyciemss_logging_wrapper
from pyciemss.utils.interface_utils import convert_to_output_format
from pyciemss.visuals import plots

EnsembleSolution = Iterable  # NOTE: [dict[str, torch.Tensor]] type argument removed because of issues with type-based dispatch.  # noqa
EnsembleInferredParameters = pyro.nn.PyroModule


def create_solution_mapping_fns(
    models: list[DynamicalSystem], solution_mappings: list[dict]
) -> list:
    solution_mapping_fs = []
    for i, model in enumerate(models):
        solution_mapping_f = interface_utils.create_mapping_function_from_observables(
            model, solution_mappings[i]
        )
        solution_mapping_fs.append(solution_mapping_f)
    return solution_mapping_fs


# TODO: create better type hint for `models`. Struggled with `Iterable[DynamicalSystem]`.
@setup_model.register(list)
@pyciemss_logging_wrapper
def setup_ensemble_model(
    models: list[DynamicalSystem],
    weights: Iterable[float],
    solution_mappings: Iterable[Callable],
    start_time: float,
    start_states: Optional[Iterable[dict[str, float]]] = None,
    *,
    total_population: float = 1.0,
    noise_model: str = "scaled_normal",
    noise_scale: float = 0.1,
    dirichlet_concentration: float = 1.0,
) -> EnsembleSystem:
    """
    Instatiate a model for a particular configuration of initial conditions
    """

    start_states = [
        {get_name(v): v.data["initial_value"] for v in model.G.variables.values()}
        for model in models
    ]

    if noise_model == "scaled_beta":
        # TODO
        noise_pseudocount = torch.as_tensor(1 / noise_scale)
        ensemble_model = copy.deepcopy(
            ScaledBetaNoiseEnsembleSystem(
                models,
                torch.as_tensor(weights) * dirichlet_concentration,
                solution_mappings,
                total_population,
                noise_pseudocount,
            )
        )
    elif noise_model == "scaled_normal":
        ensemble_model = copy.deepcopy(
            ScaledNormalNoiseEnsembleSystem(
                models,
                torch.as_tensor(weights) * dirichlet_concentration,
                solution_mappings,
                total_population,
                noise_scale,
            )
        )
    else:
        raise ValueError(f"noise_model {noise_model} not recognized")

    for i, m in enumerate(ensemble_model.models):
        start_event = StartEvent(start_time, start_states[i])
        m.load_event(start_event)
    return ensemble_model


@reset_model.register
@pyciemss_logging_wrapper
def reset_ensemble_model(ensemble: EnsembleSystem) -> EnsembleSystem:
    """
    Reset a model to its initial state.
    reset_model * setup_model = id
    """
    raise NotImplementedError


@intervene.register
@pyciemss_logging_wrapper
def intervene_ensemble_model(
    ensemble: EnsembleSystem, interventions: Iterable[Tuple[float, str, float]]
) -> EnsembleSystem:
    """
    Intervene on a model.
    """
    raise NotImplementedError


@calibrate.register
@pyciemss_logging_wrapper
def calibrate_ensemble_model(
    ensemble: EnsembleSystem,
    data: Iterable[Tuple[float, dict[str, float]]],
    num_iterations: int = 1000,
    lr: float = 0.03,
    verbose: bool = False,
    verbose_every: int = 25,
    num_particles: int = 1,
    autoguide=pyro.infer.autoguide.AutoLowRankMultivariateNormal,
    method="dopri5",
) -> EnsembleInferredParameters:
    """
    Calibrate a model. Dispatches to the calibrate method of the underlying model.
    This method is only implemented for petri net models.
    """
    # TODO: Refactor the codebase so that this can be implemented for any model that has a calibrate method.
    # This will require pulling out functions for checking the validity of the data, and for setting up the model.

    new_ensemble = copy.deepcopy(ensemble)
    observations = [
        ObservationEvent(timepoint, observation) for timepoint, observation in data
    ]

    # Again, here we assume that all observations are scaled to the first model in the ensemble.
    test_petri = new_ensemble.models[0]

    for obs in observations:
        s = 0.0
        for v in obs.observation.values():
            s += v
            assert 0 <= v <= test_petri.total_population
        assert s <= test_petri.total_population or torch.isclose(
            s, test_petri.total_population
        )

    new_ensemble.load_events(observations)

    guide = autoguide(new_ensemble)
    optim = pyro.optim.Adam({"lr": lr})
    loss = pyro.infer.Trace_ELBO(num_particles=num_particles)
    svi = pyro.infer.SVI(new_ensemble, guide, optim, loss=loss)

    pyro.clear_param_store()

    for i in range(num_iterations):
        loss = svi.step(method=method)
        if verbose:
            if i % verbose_every == 0:
                print(f"iteration {i}: loss = {loss}")

    return guide


@sample.register
@pyciemss_logging_wrapper
def sample_ensemble_model(
    ensemble: EnsembleSystem,
    timepoints: Iterable[float],
    num_samples: int,
    inferred_parameters: Optional[EnsembleInferredParameters] = None,
    *args,
    **kwargs,
) -> EnsembleSolution:
    """
    Sample from an ensemble model.
    """

    logging_events = [LoggingEvent(timepoint) for timepoint in timepoints]
    new_ensemble = copy.deepcopy(ensemble)
    new_ensemble.load_events(logging_events)
    # **kwargs is used to pass in optional model parameters, such as the solver method for an ODE.
    return Predictive(new_ensemble, guide=inferred_parameters, num_samples=num_samples)(
        *args, **kwargs
    )


@pyciemss_logging_wrapper
@prepare_interchange_dictionary.register
def prepare_interchange_dictionary(
    samples: EnsembleSolution,
    *,
    timepoints: Iterable[float],
    time_unit: Optional[str] = None,
    alpha_qs: Optional[Iterable[float]] = DEFAULT_QUANTILES,
    stacking_order: Optional[str] = "timepoints",
    visual_options: Union[None, bool, dict[str, any]] = None,
) -> dict:
    processed_samples, q_ensemble = convert_to_output_format(
        samples,
        timepoints,
        time_unit=time_unit,
        quantiles=True,
        alpha_qs=alpha_qs,
        stacking_order=stacking_order,
    )

    if visual_options:
        visual_options = {} if visual_options is True else visual_options
        schema = plots.trajectories(processed_samples, **visual_options)
        return {"data": processed_samples, "quantiles": q_ensemble, "visual": schema}
    else:
        return {"data": processed_samples, "quantiles": q_ensemble}
