import torch

from pyro.infer import Predictive
from pyro.infer.autoguide.guides import AutoNormal
from pyro.poutine import block

from causal_pyro.query.do_messenger import do

from pyciemss.ODE.abstract import ODE

from pyciemss.utils.inference_utils import run_inference
from pyciemss.utils.petri_utils import petri_to_deriv_and_observation
from pyciemss.utils.prior_utils import parse_prior

from typing import TypeVar, Iterable, Optional, Union

# Declare types
PetriNet               = TypeVar('PetriNet')
PriorJSON              = TypeVar('PriorJSON')
PriorPP                = TypeVar('PriorPP')
InferredParameters     = TypeVar('InferredParameters')
State                  = TypeVar('State')
TSpan                  = TypeVar('TSpan')
Data                   = TypeVar('Data')
InterventionSpec       = TypeVar('InterventionSpec')
Variable               = TypeVar('Variable')
ObjectiveFunction      = TypeVar('ObjectiveFunction')
Constraints            = TypeVar('Constraints')
OptimizationAlgorithm  = TypeVar('OptimizationAlgorithm')
DataCube               = TypeVar('DataCube')
OptimizationResult     = TypeVar('OptimizationResult')

def compile_pp(petri_G: PetriNet, 
               prior_json: PriorJSON) -> PriorPP:

    prior_attributes, prior_pyro_method = parse_prior(prior_json)

    PyroODE = type("PyroODE", (ODE,), prior_attributes)

    PyroODE.param_prior = prior_pyro_method

    PyroODE.deriv, PyroODE.observation_model = petri_to_deriv_and_observation(petri_G)

    init_args = []
    init_kwargs = {}

    return PyroODE(*init_args, **init_kwargs)

def sample(ode_model: PriorPP,
            num_samples: int, 
            initial_state: State, 
            tspan: TSpan,
            inferred_parameters: Optional[InferredParameters] = None) -> DataCube:
    
    '''
    Sample `num_samples` trajectories from the prior distribution over ODE models.
    '''

    return Predictive(ode_model, guide=inferred_parameters, num_samples=num_samples)(initial_state, tspan)

def infer_parameters(ode_model: PriorPP, 
                     num_iterations: int, 
                     hidden_observations: Iterable[str], 
                     data: Data,
                     initial_state: State,
                     observed_tspan: TSpan) -> InferredParameters:
    
    '''
    Use variational inference to condition `ode_model` on observed `data`.
    '''

    guide = AutoNormal(block(ode_model, hide=hidden_observations))
    run_inference(ode_model, guide, initial_state, observed_tspan, data, num_iterations=num_iterations)
    return guide

def intervene(ode_model: PriorPP, 
            intervention_spec: InterventionSpec) -> PriorPP:
    return do(ode_model, intervention_spec)

# TODO: finish optimization frontend
def optimization(initial_guess: torch.tensor,
        objective_function: ObjectiveFunction,
        constrains: Constraints,
        optimizer: OptimizationAlgorithm):
    raise NotImplementedError



