import torch
import json
import unittest

import pyro.distributions as dist

from pyciemss.ODE.models import SVIIvR, SVIIvR_simple
from pyciemss.utils import load, add_state_indicies, is_density_equal, is_intervention_density_equal
from pyciemss.ODE.frontend import compile_pp

class Petri2PyroTest(unittest.TestCase):
    """Tests for the Petri2Pyro class."""
    def setUp(self):
        """Setup for the Petri2Pyro class."""

        self.num_samples = 500

        # Total population, N.
        self.N = 100000.0
        # Initial number of infected and recovered individuals, I0 and R0.
        self.V0, self.I0, self.Iv0, self.R0 = 0., 81.0, 0., 0.
        # Everyone else, S0, is susceptible to infection initially.
        self.S0 = self.N - self.I0 - self.Iv0 - self.V0 - self.R0

        # 18 - 24 year olds
        self.I_obs = torch.tensor([81.47, 84.3, 86.44, 89.66, 93.32, 94.1, 96.31])

        self.initial_state = tuple(torch.as_tensor(s) for s in  (self.S0, self.V0, self.I0, self.Iv0, self.R0))
        self.final_observed_state = tuple(torch.as_tensor(s) for s in  (self.S0, self.V0, self.I_obs[-1], self.Iv0, self.R0))

        self.SVIIvR = SVIIvR(self.N,
                noise_prior=dist.Uniform(5., 10.),
                beta_prior=dist.Uniform(0.1, 0.3),
                betaV_prior=dist.Uniform(0.025, 0.05),
                gamma_prior=dist.Uniform(0.05, 0.35),
                gammaV_prior=dist.Uniform(0.1, 0.4),
                nu_prior=dist.Uniform(0.001, 0.01))


    def test_petri2pyro(self):
        """Tests for the Petri2Pyro class."""
        # Setup Parameters
        
        prior_path = "./test/models/SVIIvR_simple/prior.json"
        petri_path = "./test/models/SVIIvR_simple/petri.json"

        with open(prior_path) as f:
            prior_json = json.load(f)

        petri_G = load(petri_path)
        petri_G = add_state_indicies(petri_G)

        model_compiled = compile_pp(petri_G, prior_json)
        model = SVIIvR_simple

        num_samples = 100

        self.assertFalse(is_density_equal(model, model_compiled, num_samples=num_samples))
        self.assertFalse(is_intervention_density_equal(model, model_compiled, intervention={'x': 1}, num_samples=num_samples))