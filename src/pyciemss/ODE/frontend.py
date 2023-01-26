from pyciemss.ODE.abstract import ODE

import pyro
import pyro.distributions as dist

from pyro.nn import pyro_method

from pyciemss.utils.petri_utils import petri_to_deriv_and_observation

# Keys here are names in the JSON template, values are the pyro distribution.
PRIOR_MAPPING = {"Uniform" : dist.Uniform, 
                 "Normal" : dist.Normal}

def prior_annotate(name):
    return "{}_prior".format(name)

def parse_prior(prior_json):

    prior_attributes = {}

    for key, val in prior_json.items():
        # First element in the list is the distribution type. All remaining
        # elements are the distribution arguments, hence the splatting below.
        prior_attributes[prior_annotate(key)] = PRIOR_MAPPING[val[0]](*val[1:])

    @pyro_method
    def prior_pyro_method(self):
        '''
        TODO: document
        '''
        for key in prior_json.keys():
            # Programmatic represention of `self.var = pyro.sample("var", self.var_prior)`
            setattr(self, key, pyro.sample(key, getattr(self, prior_annotate(key))))

    return prior_attributes, prior_pyro_method


def compile_pp(petri_G, prior_json):

    prior_attributes, prior_pyro_method = parse_prior(prior_json)

    PyroODE = type("PyroODE", (ODE,), prior_attributes)

    PyroODE.param_prior = prior_pyro_method

    PyroODE.deriv, PyroODE.observation_model = petri_to_deriv_and_observation(petri_G)

    init_args = []
    init_kwargs = {}

    return PyroODE(*init_args, **init_kwargs)