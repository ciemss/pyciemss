import unittest

import torch
import json

from pyciemss.utils import load, add_state_indicies, get_tspan
from pyciemss.ODE.interventions import constant_intervention

from pyciemss.ODE.frontend import compile_pp, sample, infer_parameters, intervene

class TestDensityTest(unittest.TestCase):
    """Tests for the high level frontend methods."""

    def test_december_demo_walkthrough(self):
        """Test all of the frontend methods on the december demo."""

        # External inputs
        PRIOR_PATH = "test/models/SVIIvR_simple/prior.json"
        PETRI_PATH = "test/models/SVIIvR_simple/petri.json"

        with open(PRIOR_PATH) as f:
            prior_json = json.load(f)

        petri_G = load(PETRI_PATH)
        petri_G = add_state_indicies(petri_G)

        ode_model = compile_pp(petri_G, prior_json)
        self.assertIsNotNone(ode_model)

        # External inputs
        NUM_SAMPLES = 3
        INITIAL_STATE = tuple(torch.as_tensor(s) for s in (1 - 81/100000, 0., 81/100000, 0., 0.))
        TSPAN = get_tspan(1, 99, 99)

        prior_data_cube = sample(ode_model,
                                 NUM_SAMPLES,
                                 INITIAL_STATE,
                                 TSPAN)
        self.assertIsNotNone(prior_data_cube)

        # External inputs
        NUM_ITERATIONS = 2
        HIDDEN_OBSERVATIONS = ["S_obs", "V_obs", "Iv_obs", "R_obs"]
        DATA = {"I_obs" : torch.tensor([81.47, 84.3, 86.44, 89.66, 93.32, 94.1, 96.31])/100000,
                "S_obs" : None,
                "V_obs" : None,
                "Iv_obs": None,
                "R_obs" : None}
        OBSERVED_TSPAN = get_tspan(1, 7, 7)

        inferred_parameters = infer_parameters(ode_model,
                                               NUM_ITERATIONS,
                                               HIDDEN_OBSERVATIONS,
                                               DATA,
                                               INITIAL_STATE,
                                               OBSERVED_TSPAN
                                               )
        self.assertIsNotNone(inferred_parameters)

        # External inputs
        INTERVENTION = constant_intervention("SV_flux", torch.tensor(0.), TSPAN)

        intervened_ode_model = intervene(ode_model, INTERVENTION)
        self.assertIsNotNone(intervened_ode_model)

        
        
