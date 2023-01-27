from pyciemss.ODE.abstract import ODE

from pyciemss.utils.petri_utils import petri_to_deriv_and_observation
from pyciemss.utils.prior_utils import parse_prior

from typing import Dict, Tuple, TypeVar, Optional, Tuple, Union

# Declare types
PetriNet               = TypeVar('PetriNet')
PriorJSON              = TypeVar('PriorJSON')
PriorPP                = TypeVar('PriorPP')
VariationalPosteriorPP = TypeVar('VariationalPosteriorPP')
PP                     = Union[PriorPP, VariationalPosteriorPP]
Data                   = TypeVar('Data')
Intervention           = TypeVar('Intervention')
Variable               = TypeVar('Variable')
OptimizationAlgorithm  = TypeVar('OptimizationAlgorithm')
DataCube               = TypeVar('DataCube')

def compile_pp(petri_G: PetriNet, prior_json: PriorJSON) -> PriorPP:

    prior_attributes, prior_pyro_method = parse_prior(prior_json)

    PyroODE = type("PyroODE", (ODE,), prior_attributes)

    PyroODE.param_prior = prior_pyro_method

    PyroODE.deriv, PyroODE.observation_model = petri_to_deriv_and_observation(petri_G)

    init_args = []
    init_kwargs = {}

    return PyroODE(*init_args, **init_kwargs)

def vi_condition(ode_model: PriorPP, data: Data) -> VariationalPosteriorPP:
    raise NotImplementedError # TODO

def sample(ode_model: PP):
    raise NotImplementedError # TODO

# TODO: make the type signature more refined. PriorPP -> PriorPP, etc.
def intervene(ode_model: PP, intervention: Intervention) -> PP:
    raise NotImplementedError # TODO

#TODO: wait until we merge UT code.
def ouu():
    raise NotImplementedError



