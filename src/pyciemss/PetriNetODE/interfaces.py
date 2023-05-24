import copy
from typing import Iterable, Optional, Tuple, Union, cast

import mira
import pyro
import torch
from pyro.infer import Predictive
from pyro.nn import PyroModule
from torch import Tensor
from typing_extensions import TypeAlias

# Load base interfaces
from pyciemss.interfaces import (
    calibrate,
    intervene,
    optimize,
    reset_model,
    sample,
    setup_model,
)
from pyciemss.PetriNetODE.base import (
    MiraPetriNetODESystem,
    PetriNetODESystem,
    ScaledBetaNoisePetriNetODESystem,
)
from pyciemss.PetriNetODE.events import (
    Event,
    LoggingEvent,
    ObservationEvent,
    StartEvent,
    StaticParameterInterventionEvent,
)

# from pyciemss.risk.ouu import solveOUU

# TODO: These interfaces should probably be just in terms of JSON-like objects.

PetriSolution = dict[str, Tensor]
PetriInferredParameters: TypeAlias = PyroModule


def load_petri_model(
    petri_model_or_path: Union[str, mira.metamodel.TemplateModel, mira.modeling.Model],
    add_uncertainty=True,
    pseudocount=1.0,
) -> PetriNetODESystem:
    """
    Load a petri net from a file and compile it into a probabilistic program.
    """

    if add_uncertainty:
        model = ScaledBetaNoisePetriNetODESystem.from_mira(petri_model_or_path)
        model.pseudocount = torch.tensor(pseudocount)
        return model
    else:
        return MiraPetriNetODESystem.from_mira(petri_model_or_path)


@setup_model.register
def setup_petri_model(
    petri: PetriNetODESystem,
    start_time: float,
    start_state: dict[str, float],
) -> PetriNetODESystem:
    """
    Instatiate a model for a particular configuration of initial conditions
    """
    # TODO: Figure out how to do this without copying the petri net.
    start_event = StartEvent(start_time, start_state)
    new_petri = copy.deepcopy(petri)
    new_petri.load_event(start_event)
    return new_petri


@reset_model.register
def reset_petri_model(petri: PetriNetODESystem) -> PetriNetODESystem:
    """
    Reset a model to its initial state.
    reset_model * setup_model = id
    """
    new_petri = copy.deepcopy(petri)
    new_petri.reset()
    return new_petri


@intervene.register
def intervene_petri_model(
    petri: PetriNetODESystem, interventions: Iterable[Tuple[float, str, float]]
) -> PetriNetODESystem:
    """
    Intervene on a model.
    """
    # Note: this will have to change if we want to add more sophisticated interventions.
    intervention_events = [
        StaticParameterInterventionEvent(timepoint, parameter, value)
        for timepoint, parameter, value in interventions
    ]
    new_petri = copy.deepcopy(petri)
    cast_intervention_events: Iterable[Event] = cast(
        Iterable[Event], intervention_events
    )
    new_petri.load_events(cast_intervention_events)
    return new_petri


@calibrate.register
def calibrate_petri(
    petri: PetriNetODESystem,
    data: Iterable[Tuple[float, dict[str, float]]],
    num_iterations: int = 1000,
    lr: float = 0.03,
    verbose: bool = False,
    num_particles: int = 1,
    autoguide=pyro.infer.autoguide.AutoLowRankMultivariateNormal,
    method="dopri5",
) -> PetriInferredParameters:
    """
    Use variational inference with a mean-field variational family to infer the parameters of the model.
    """
    new_petri = copy.deepcopy(petri)
    observations = [
        ObservationEvent(timepoint, observation) for timepoint, observation in data
    ]

    for obs in observations:
        s = torch.tensor(0.0)
        for v in obs.observation.values():
            s += v
            assert 0 <= v <= petri.total_population
        assert s <= petri.total_population or torch.isclose(s, petri.total_population)
    new_petri.load_events(observations)

    guide = autoguide(new_petri)
    optim = pyro.optim.Adam({"lr": lr})
    loss = pyro.infer.Trace_ELBO(num_particles=num_particles)
    svi = pyro.infer.SVI(new_petri, guide, optim, loss=loss)

    pyro.clear_param_store()

    for i in range(num_iterations):
        loss = svi.step(method=method)
        if verbose:
            if i % 25 == 0:
                print(f"iteration {i}: loss = {loss}")

    return guide


@sample.register
def sample_petri(
    petri: PetriNetODESystem,
    timepoints: Iterable[float],
    num_samples: int,
    inferred_parameters: Optional[PetriInferredParameters] = None,
    method="dopri5",
) -> PetriSolution:
    """
    Sample `num_samples` trajectories from the prior or posterior distribution over ODE models.
    """
    assert hasattr(
        petri, "_static_events"
    ), "Please initialize the model before sampling."
    assert len(petri._static_events) > 0, "No events initialized."
    assert isinstance(
        petri._static_events[0], StartEvent
    ), "First event should be a StartEvent."
    logging_events = [LoggingEvent(timepoint) for timepoint in timepoints]
    new_petri = copy.deepcopy(petri)
    new_petri.load_events(logging_events)
    return Predictive(new_petri, guide=inferred_parameters, num_samples=num_samples)(
        method=method
    )


@optimize.register
def optimize_petri(
    petri: PetriNetODESystem, initial_guess, objective_function, constraints, optimizer
):
    # TODO: This probably won't work out of the box. Will need to work with Anirban to refactor this.
    #        return solveOUU(initial_guess,
    #                    objective_function,
    #                    constraints,
    #                    optimizer_algorithm=optimizer).solve()
    raise NotImplementedError
