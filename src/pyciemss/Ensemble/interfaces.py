import copy
from typing import Callable, Optional, Sequence, Tuple

import pyro
import torch
from pyro import poutine
from pyro.infer import Predictive
from typing_extensions import TypeAlias

from pyciemss.Ensemble.base import EnsembleSystem, ScaledBetaNoiseEnsembleSystem
from pyciemss.interfaces import (
    DynamicalSystem,
    calibrate,
    intervene,
    optimize,
    reset_model,
    sample,
    setup_model,
)

# TODO: probably refactor this out later.
from pyciemss.PetriNetODE.events import (
    LoggingEvent,
    ObservationEvent,
    StartEvent,
    StaticParameterInterventionEvent,
)

EnsembleSolution = Sequence[dict[str, torch.Tensor]]
EnsembleInferredParameters: TypeAlias = pyro.nn.PyroModule


# TODO: create better type hint for `models`. Struggled with `Iterable[DynamicalSystem]`.
@setup_model.register(list)  # type: ignore
def setup_ensemble_model(
    models: list[DynamicalSystem],
    weights: Sequence[float],
    solution_mappings: Sequence[Callable],
    start_time: float,
    start_states: Sequence[dict[str, float]],
    total_population: float = 1.0,
    noise_pseudocount: float = 1.0,
    dirichlet_concentration: float = 1.0,
) -> EnsembleSystem:
    """
    Instatiate a model for a particular configuration of initial conditions
    """
    ensemble_model = copy.deepcopy(
        ScaledBetaNoiseEnsembleSystem(
            models,
            torch.as_tensor(weights) * dirichlet_concentration,
            solution_mappings,
            total_population,
            noise_pseudocount,
        )
    )
    for i, m in enumerate(ensemble_model.models):
        start_event = StartEvent(start_time, start_states[i])
        m.load_event(start_event)
    return ensemble_model


@reset_model.register
def reset_ensemble_model(ensemble: EnsembleSystem) -> EnsembleSystem:
    """
    Reset a model to its initial state.
    reset_model * setup_model = id
    """
    raise NotImplementedError


@intervene.register
def intervene_ensemble_model(
    ensemble: EnsembleSystem, interventions: Sequence[Tuple[float, str, float]]
) -> EnsembleSystem:
    """
    Intervene on a model.
    """
    raise NotImplementedError


@calibrate.register
def calibrate_ensemble_model(
    ensemble: EnsembleSystem,
    data: Sequence[Tuple[float, dict[str, float]]],
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
        s = torch.tensor(0.0)
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
def sample_ensemble_model(
    ensemble: EnsembleSystem,
    timepoints: Sequence[float],
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
