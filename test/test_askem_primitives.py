import unittest

import torch
import json

from pyciemss.utils import load, add_state_indicies, get_tspan
from pyciemss.ODE.interventions import constant_intervention

from pyciemss.ODE.askem_primitives import sample, infer_parameters, intervene, optimization
from pyciemss.ODE.models import SVIIvR

from pyciemss.risk.ouu import RandomDisplacementBounds, computeRisk
from pyciemss.risk.qoi import nday_rolling_average
from pyciemss.risk.risk_measures import alpha_superquantile, alpha_quantile

class TestDensityTest(unittest.TestCase):
    """Tests for the high level frontend methods."""

    def test_december_demo_walkthrough(self):
        """Test all of the frontend methods on the december demo."""

        ode_model = SVIIvR(N=1)
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

        # Define problem specifics.
        RISK_BOUND = 250.
        X_MIN = 0.
        X_MAX = 0.95

        N_SAMPLES = int(5)

        RISK_ALPHA = 0.9

        # Control action / intervention.
        INITIAL_GUESS = 0.75
        INTERVENTION = lambda x: constant_intervention("nu", x, TSPAN)

        # Objective function.
        OBJECTIVE_FUNCTION = lambda x: x  # minimize the scalar value itself.

        # Define the risk measure.
        VAX_RISK = computeRisk(
                                model=ode_model,
                                intervention_fun=INTERVENTION,
                                qoi=lambda y: nday_rolling_average(y, contexts=["I_obs"]),
                                model_state=INITIAL_STATE,
                                tspan=TSPAN,
                                risk_measure=lambda z: alpha_superquantile(z, alpha=RISK_ALPHA),
                                num_samples=N_SAMPLES,
                                guide=inferred_parameters,
                                )

        # Define problem constraints.
        CONSTRAINTS = (
                        # risk constraint
                        {'type': 'ineq', 'fun': lambda x: RISK_BOUND - VAX_RISK(x)},

                        # bounds on control
                        # NOTE: perhaps use scipy.optimize.LinearConstraint in the future
                        {'type': 'ineq', 'fun': lambda x: x - X_MIN},
                        {'type': 'ineq', 'fun': lambda x: X_MAX - x}
                    )

        OPTIMIZER = "COBYLA"

        solution = optimization(INITIAL_GUESS, OBJECTIVE_FUNCTION, CONSTRAINTS, OPTIMIZER)
        self.assertIsNotNone(solution)