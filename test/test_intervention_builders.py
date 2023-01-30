import unittest

import torch
from pyciemss.utils import get_tspan
from pyciemss.ODE.interventions import constant_intervention_builder, parameter_intervention_builder, state_dependent_intervention_builder, time_and_state_dependent_intervention_builder, time_dependent_intervention_builder

from pyciemss.ODE.models import SVIIvR
from pyciemss.ODE.askem_primitives import sample, intervene

class TestInterventions(unittest.TestCase):
    '''
    Test for the intervention builders
    '''
    def setUp(self):
        
        # Total population, N.
        self.N = 100000.0
        
        # Initial number of infected and recovered individuals, I0 and R0.
        self.V0, self.I0, self.Iv0, self.R0 = 0., 81.0, 0., 0.
        
        # Everyone else, S0, is susceptible to infection initially.
        self.S0 = self.N - self.I0 - self.Iv0 - self.V0 - self.R0

        self.initial_state = tuple(torch.as_tensor(s) for s in  (self.S0, self.V0, self.I0, self.Iv0, self.R0))
        self.tspan = get_tspan(1, 89, 89)

        self.model = SVIIvR(self.N)
        self.num_samples = 10


    def test_constant_intervention_builder(self):
        # Just testing whether the intervention builder constructs a model
        intervention = constant_intervention_builder("SV_flux", torch.tensor([1.]), self.tspan)
        intervened_model = intervene(self.model, intervention)
        result = sample(intervened_model, self.num_samples, self.initial_state, self.tspan)
        self.assertIsNotNone(result)

    def test_parameter_intervention_builder(self):
        # Just testing whether the intervention builder constructs a model
        intervention = parameter_intervention_builder("nu", torch.tensor([1.]))
        intervened_model = intervene(self.model, intervention)
        result = sample(intervened_model, self.num_samples, self.initial_state, self.tspan)
        self.assertIsNotNone(result)
    
    def test_state_dependent_intervention_builder(self):
        # Just testing whether the intervention builder constructs a model
        intervention = state_dependent_intervention_builder("SV_flux", lambda x: x * 0.3, self.tspan)
        intervened_model = intervene(self.model, intervention)
        result = sample(intervened_model, self.num_samples, self.initial_state, self.tspan)
        self.assertIsNotNone(result)

    def test_time_dependent_intervention_builder(self):
        # Just testing whether the intervention builder constructs a model
        intervention = time_dependent_intervention_builder("SV_flux", lambda t: 1 / t, self.tspan)
        intervened_model = intervene(self.model, intervention)
        result = sample(intervened_model, self.num_samples, self.initial_state, self.tspan)
        self.assertIsNotNone(result)

    def test_time_and_state_dependent_intervention_builder(self):
        # Just testing whether the intervention builder constructs a model
        intervention = time_and_state_dependent_intervention_builder("SV_flux", lambda t, x: x / t, self.tspan)
        intervened_model = intervene(self.model, intervention)
        result = sample(intervened_model, self.num_samples, self.initial_state, self.tspan)
        self.assertIsNotNone(result)    