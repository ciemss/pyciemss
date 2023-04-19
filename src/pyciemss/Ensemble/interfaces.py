import pyro
import torch
from pyro.infer import Predictive

from pyciemss.interfaces import setup_model, reset_model, intervene, sample, calibrate, optimize, DynamicalSystem

from pyciemss.Ensemble.base import EnsembleSystem

from typing import Iterable, Optional, Tuple
import copy

# TODO: probably refactor this out later.
from pyciemss.PetriNetODE.events import StartEvent, ObservationEvent, LoggingEvent, StaticParameterInterventionEvent

EnsembleSolution = Iterable[dict[str, torch.Tensor]]
EnsembleInferredParameters = pyro.nn.PyroModule

# TODO: create better type hint for `models`. Struggled with `Iterable[DynamicalSystem]`.
@setup_model.register(list)
def setup_ensemble_model(models: list[DynamicalSystem], 
                         weights: Iterable[float], 
                         start_time: float,
                         start_states: Iterable[dict[str, float]],
                         ) -> EnsembleSystem:
    '''
    Instatiate a model for a particular configuration of initial conditions
    '''
    ensemble_model = copy.deepcopy(EnsembleSystem(models, torch.as_tensor(weights)))
    for i, m in enumerate(ensemble_model.models):
        start_event = StartEvent(start_time, start_states[i])
        m.load_event(start_event)
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
    raise NotImplementedError
    # new_ensemble = copy.deepcopy(ensemble)
    # new_ensemble.intervene(interventions)
    # return new_ensemble

@calibrate.register
def calibrate_ensemble_model(ensemble: EnsembleSystem, observations: Iterable[ObservationEvent]) -> EnsembleInferredParameters:
    raise NotImplementedError

@sample.register
def sample_ensemble_model(ensemble: EnsembleSystem,
                          timepoints: Iterable[float],
                          num_samples: int,
                          inferred_parameters: Optional[EnsembleInferredParameters] = None,
                          *args,
                          **kwargs) -> EnsembleSolution:
    '''
    Sample from an ensemble model.
    '''

    logging_events = [LoggingEvent(timepoint) for timepoint in timepoints]
    new_ensemble = copy.deepcopy(ensemble)
    new_ensemble.load_events(logging_events)
    # **kwargs is used to pass in optional model parameters, such as the solver method for an ODE.
    return Predictive(new_ensemble, guide=inferred_parameters, num_samples=num_samples)(*args, **kwargs)
