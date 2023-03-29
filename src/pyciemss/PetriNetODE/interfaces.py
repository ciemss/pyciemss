import pyro
import torch
from pyro.infer import Predictive

from pyciemss.PetriNetODE.base import ODE, BetaNoisePetriNetODESystem, PetriNetODESystem
from pyciemss.risk.ouu import solveOUU

from typing import Iterable, Optional, Tuple
import functools
import copy

# Load base interfaces
from pyciemss.interfaces import setup_model, reset_model, intervene, sample, calibrate, optimize

from pyciemss.PetriNetODE.events import StartEvent, ObservationEvent, LoggingEvent, StaticParameterInterventionEvent

# TODO: These interfaces should probably be just in terms of JSON-like objects.

PetriSolution = dict[str, torch.Tensor]
PetriInferredParameters = pyro.nn.PyroModule

@functools.singledispatch
def load_petri_model(model_path_or_petri, *args, **kwargs) -> ODE:
    '''
    Load a petri net from a file and/or compile it into a probabilistic program.
    '''
    raise NotImplementedError

@load_petri_model.register
def load_model_from_path(petri_path: str, add_uncertainty=True) -> ODE:
    '''
    Load a petri net from a file and compile it into a probabilistic program.
    '''
    if add_uncertainty:
        return BetaNoisePetriNetODESystem.from_mira(petri_path)
    else:
        return PetriNetODESystem.from_mira(petri_path)

@load_petri_model.register
def load_model_from_petri(petri, *args, **kwargs) -> ODE:
    '''
    Compile a petri net into a probabilistic program.
    '''
    # TODO: load from a Mira object directly.
    raise NotImplementedError

@setup_model.register
def setup_petri_model(petri: ODE, 
                      start_event: StartEvent,
                    ) -> ODE:    
    '''
    Instatiate a model for a particular configuration of initial conditions
    '''
    # TODO: Figure out how to do this without copying the petri net.
    new_petri = copy.deepcopy(petri)
    new_petri.load_event(start_event)
    return new_petri

@reset_model.register
def reset_petri_model(petri: ODE) -> ODE:
    '''
    Reset a model to its initial state.
    reset_model * setup_model = id
    '''
    new_petri = copy.deepcopy(petri)
    new_petri.reset()
    return new_petri

@intervene.register
def intervene_petri_model(petri: ODE, interventions: Iterable[StaticParameterInterventionEvent]) -> ODE:
    '''
    Intervene on a model.
    '''
    new_petri = copy.deepcopy(petri)
    new_petri.load_events(interventions)
    return new_petri

@calibrate.register
def calibrate_petri(petri: ODE, 
                    observations: Iterable[ObservationEvent],
                    num_iterations: int = 1000, 
                    lr: float = 0.03, 
                    verbose: bool = False,
                    num_particles: int = 1,
                    ) -> PetriInferredParameters:
    
    '''
    Use variational inference with a mean-field variational family to infer the parameters of the model.
    '''
    new_petri = copy.deepcopy(petri)
    new_petri.load_events(observations)

    guide = pyro.infer.autoguide.AutoNormal(new_petri)
    optim = pyro.optim.Adam({"lr": lr})
    loss = pyro.infer.Trace_ELBO(num_particles=num_particles)
    svi = pyro.infer.SVI(new_petri, guide, optim, loss=loss)

    pyro.clear_param_store()

    for i in range(num_iterations):
        loss = svi.step()
        if verbose:
            if i % 25 == 0:
                print(f"iteration {i}: loss = {loss}")
    
    return guide

@sample.register
def sample_petri(petri:ODE,
                 timepoints: Iterable[float],
                 num_samples: int,
                 inferred_parameters: Optional[PetriInferredParameters] = None) -> PetriSolution:
    
    '''
    Sample `num_samples` trajectories from the prior or posterior distribution over ODE models.
    '''
    logging_events = [LoggingEvent(timepoint) for timepoint in timepoints]
    new_petri = copy.deepcopy(petri)
    new_petri.load_events(logging_events)
    return Predictive(new_petri, guide=inferred_parameters, num_samples=num_samples)()

@optimize.register
def optimize_petri(petri:ODE,
                   initial_guess,
                   objective_function,
                   constraints,
                   optimizer):
        # TODO: This probably won't work out of the box. Will need to work with Anirban to refactor this.
        return solveOUU(initial_guess,
                    objective_function,
                    constraints,
                    optimizer_algorithm=optimizer).solve()