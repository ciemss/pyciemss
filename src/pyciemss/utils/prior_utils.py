import pyro
import pyro.distributions as dist

from pyro.nn import pyro_method

def prior_annotate(name):
    return "{}_prior".format(name)

def parse_prior(prior_json):

    # TODO: make noise model more flexible
    assert "noise_var" in prior_json.keys()

    prior_attributes = {}

    for key, val in prior_json.items():
        # First element in the list is the distribution type. All remaining
        # elements are the distribution arguments, hence the splatting below.
        prior_attributes[prior_annotate(key)] = getattr(dist, val[0])(*val[1:])

    @pyro_method
    def prior_pyro_method(self):
        '''
        Sample parameters from their prior distributions.
        '''
        for key in prior_json.keys():
            # Programmatic represention of `self.var = pyro.sample("var", self.var_prior)`
            setattr(self, key, pyro.sample(key, getattr(self, prior_annotate(key))))

    return prior_attributes, prior_pyro_method