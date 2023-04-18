import pyro
import torch
from pyro.infer import Predictive

from pyciemss.interfaces import setup_model, reset_model, intervene, sample, calibrate, optimize, DynamicalSystem

from pyciemss.Ensemble.base import EnsembleSystem

from typing import Iterable, Optional
import copy

# TODO: probably refactor this out later.
from pyciemss.PetriNetODE.events import StartEvent, ObservationEvent, LoggingEvent, StaticParameterInterventionEvent

EnsembleSolution = Iterable[dict[str, torch.Tensor]]
EnsembleInferredParameters = pyro.nn.PyroModule

@setup_model.register
def setup_ensemble_model(models: Iterable[DynamicalSystem], weights: Iterable[float], event: StartEvent) -> EnsembleSystem:
    '''
    Instatiate a model for a particular configuration of initial conditions
    '''
    start_event = StartEvent(event.start_time, event.start_state)
    ensemble_model = copy.deepcopy(EnsembleSystem(models, weights))
    ensemble_model.load_event(start_event)
    return ensemble_model

@reset_model.register
def reset_ensemble_model(ensemble: EnsembleSystem) -> EnsembleSystem:
    '''
    Reset a model to its initial state.
    reset_model * setup_model = id
    '''
    new_ensemble = copy.deepcopy(ensemble)
    new_ensemble.reset()
    return new_ensemble

@intervene.register
def intervene_ensemble_model(ensemble: EnsembleSystem, interventions: Iterable[Tuple[float, str, float]]) -> EnsembleSystem:
    '''
    Intervene on a model.
    '''
    new_ensemble = copy.deepcopy(ensemble)
    new_ensemble.intervene(interventions)
    return new_ensemble

@calibrate.register
def calibrate_ensemble_model(ensemble: EnsembleSystem, observations: Iterable[ObservationEvent]) -> EnsembleInferredParameters:
    raise NotImplementedError

@sample.register
def sample_ensemble_model(ensemble: EnsembleSystem,
                          timepoints: Iterable[float],
                          num_samples: int,
                          inferred_parameters: Optional[EnsembleInferredParameters] = None,
                          **kwargs) -> EnsembleSolution:
    '''
    Sample from an ensemble model.
    '''

    logging_events = [LoggingEvent(timepoint) for timepoint in timepoints]
    new_ensemble = copy.deepcopy(ensemble)
    new_ensemble.load_events(logging_events)
    # **kwargs is used to pass in optional model parameters, such as the solver method for an ODE.
    return Predictive(new_ensemble, guide=inferred_parameters, num_samples=num_samples)(**kwargs)
