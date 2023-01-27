from pyciemss.ODE.abstract import ODE

import pyro
import pyro.distributions as dist

from pyro.nn import pyro_method

from pyciemss.utils.petri_utils import petri_to_deriv_and_observation
from pyciemss.utils.prior_utils import parse_prior

def compile_pp(petri_G, prior_json):

    prior_attributes, prior_pyro_method = parse_prior(prior_json)

    PyroODE = type("PyroODE", (ODE,), prior_attributes)

    PyroODE.param_prior = prior_pyro_method

    PyroODE.deriv, PyroODE.observation_model = petri_to_deriv_and_observation(petri_G)

    init_args = []
    init_kwargs = {}

    return PyroODE(*init_args, **init_kwargs)

# TODO: Add types and fill out arguments

def vi_condition(ode_model, data):
    raise NotImplementedError # TODO

def sample(ode_model):
    raise NotImplementedError # TODO

def intervention_builder(intervention_spec):
    raise NotImplementedError # TODO

def intervene(ode_model, intervention):
    raise NotImplementedError # TODO

def ouu():
    raise NotImplementedError # TODO



