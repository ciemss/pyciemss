import unittest

import torch
import pyro.distributions as dist
from pyro.infer import Predictive
import numpy as np

from pyciemss.utils import get_tspan
from pyciemss.risk.ouu import computeRisk
from pyciemss.risk.qoi import threshold_exceedance
from pyciemss.risk.risk_measures import alpha_quantile
from pyciemss.ODE.models import SVIIvR
from pyciemss.ODE.interventions import constant_intervention_builder


class TestOUU(unittest.TestCase):
    """
    Test for the risk-based OUU methods and classes.
    """

    def setUp(self):

        # Total population, N.
        self.N = 100000.0

        # Initial number of infected and recovered individuals, I0 and R0.
        self.V0, self.I0, self.Iv0, self.R0 = 0.0, 81.0, 0.0, 0.0

        # Everyone else, S0, is susceptible to infection initially.
        self.S0 = self.N - self.I0 - self.Iv0 - self.V0 - self.R0

        self.I_obs = torch.tensor([81.47, 84.3, 86.44, 89.66, 93.32, 94.1, 96.31])

        self.initial_state = tuple(
            torch.as_tensor(s) for s in (self.S0, self.V0, self.I0, self.Iv0, self.R0)
        )
        self.tspan = get_tspan(1, 89, 89)

        self.model = SVIIvR(
            self.N,
            noise_prior=dist.Uniform(5.0, 10.0),
            beta_prior=dist.Uniform(0.1, 0.3),
            betaV_prior=dist.Uniform(0.025, 0.05),
            gamma_prior=dist.Uniform(0.05, 0.35),
            gammaV_prior=dist.Uniform(0.1, 0.4),
            nu_prior=dist.Uniform(0.001, 0.01),
        )

        self.N_SAMPLES = 10

        torch.manual_seed(42)
        self.dataCube = Predictive(self.model, num_samples=10)(
            self.initial_state, self.tspan
        )

    def test_computeRisk(self):
        self.Risk = computeRisk(
            model=self.model,
            intervention_fun=lambda x: constant_intervention_builder(
                "nu", x, self.tspan
            ),
            qoi=lambda y: threshold_exceedance(y, threshold=1000.0, contexts=["I_obs"]),
            model_state=self.initial_state,
            tspan=self.tspan,
            risk_measure=alpha_quantile,
            num_samples=self.N_SAMPLES,
        )

        torch.manual_seed(0)
        actual = self.Risk(0.05)
        expected = 1.0
        self.assertEqual(actual, expected)
