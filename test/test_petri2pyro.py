import torch
import json
import unittest

import pyro.distributions as dist

import os

print(os.getcwd())

import sys
sys.path.append("/src/")

from pyciemss.ODE.models import SVIIvR, SVIIvR_simple
from pyciemss.utils import load, add_state_indicies, is_density_equal, is_intervention_density_equal, get_tspan
from pyciemss.ODE.frontend import compile_pp
from pyciemss.ODE.interventions import constant_intervention, time_dependent_intervention, state_dependent_intervention, parameter_intervention

class Petri2PyroTest(unittest.TestCase):
    """Tests for the Petri2Pyro class."""
    def setUp(self):
        """Setup for the Petri2Pyro class."""

        self.num_samples = 5

        # Total population, N.
        self.N = 1.
        # Initial number of infected and recovered individuals, I0 and R0.
        self.V0, self.I0, self.Iv0, self.R0 = 0., 0.001, 0., 0.
        # Everyone else, S0, is susceptible to infection initially.
        self.S0 = self.N - self.I0 - self.Iv0 - self.V0 - self.R0

        self.initial_state = tuple(torch.as_tensor(s) for s in  (self.S0, self.V0, self.I0, self.Iv0, self.R0))
        self.tspan = get_tspan(1, 100, 100)

    def test_petri2pyro(self):
        """Tests for the Petri2Pyro class."""
        # Setup Parameters
        
        prior_path = "test/models/SVIIvR_simple/prior.json"
        petri_path = "test/models/SVIIvR_simple/petri.json"

        with open(prior_path) as f:
            prior_json = json.load(f)

        petri_G = load(petri_path)
        petri_G = add_state_indicies(petri_G)

        model_compiled = compile_pp(petri_G, prior_json)
        model = SVIIvR_simple()

        intervention = constant_intervention("SV_flux", torch.tensor(0.0), self.tspan)

        model_args = (self.initial_state, self.tspan)

        self.assertTrue(is_density_equal(model, model_compiled, *model_args))
        self.assertTrue(is_intervention_density_equal(model, model_compiled, intervention, *model_args))