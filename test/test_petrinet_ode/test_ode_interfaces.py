import unittest
import os

from mira.examples.sir import sir_parameterized as sir

import torch

from pyciemss.PetriNetODE.interfaces import load_petri_model, setup_model, reset_model, intervene, sample, calibrate, optimize
from pyciemss.risk.qoi import scenario2dec_nday_average

class TestODEInterfaces(unittest.TestCase):
    '''Tests for the ODE interfaces.'''
    
    # Setup for the tests
    def setUp(self):
        MIRA_PATH = "test/models/evaluation_examples/scenario_1/"

        filename = "scenario1_sir_mira.json"
        self.filename = os.path.join(MIRA_PATH, filename)
        self.initial_time = 0.0
        self.initial_state = {"susceptible_population": 0.99, "infected_population": 0.01, "immune_population": 0.0}

    def test_load_petri_from_file(self):
        '''Test the load_petri function when called on a string.'''
        model = load_petri_model(self.filename, add_uncertainty=True)
        self.assertIsNotNone(model)

        model = load_petri_model(self.filename, add_uncertainty=False)
        self.assertIsNotNone(model)

    def test_load_petri_from_mira_registry(self):
        '''Test the load_petri function when called on a mira.modeling.Model'''
        model = load_petri_model(sir, add_uncertainty=True)
        self.assertIsNotNone(model)

        model = load_petri_model(sir, add_uncertainty=False)
        self.assertIsNotNone(model)
    
    def test_setup_model(self):
        '''Test the setup_model function.'''
        for model in [load_petri_model(self.filename), 
                      load_petri_model(self.filename, pseudocount=2.0)]:
            new_model = setup_model(model, self.initial_time, self.initial_state)

            self.assertIsNotNone(new_model)
            self.assertEqual(len(new_model._static_events), 1)

            # Check that setup_model is not inplace.
            self.assertEqual(len(model._static_events), 0)
        
    def test_reset_model(self):
        '''Test the reset_model function.'''
        
        model = load_petri_model(self.filename)
        model = setup_model(model, self.initial_time, self.initial_state)
        self.assertEqual(len(model._static_events), 1)
        
        new_model = reset_model(model)
        self.assertEqual(len(new_model._static_events), 0)

        # Check that reset_model is not inplace.
        self.assertEqual(len(model._static_events), 1)

    def test_intervene(self):
        '''Test the intervene function.'''
        model = load_petri_model(self.filename)
        model = setup_model(model, self.initial_time, self.initial_state)
        
        t = 0.2
        intervened_parameter = "beta"
        new_value = 0.5

        new_model = intervene(model, [(t, intervened_parameter, new_value)])
        
        self.assertEqual(len(new_model._static_events), 2)

        # Check that intervene is not inplace.
        self.assertEqual(len(model._static_events), 1)

    def test_calibrate(self):
        '''Test the calibrate function.'''
        model = load_petri_model(self.filename)
        model = setup_model(model, self.initial_time, self.initial_state)
        
        data = [(0.2, {"infected_population": 0.1}), (0.4, {"infected_population": 0.2}), (0.6, {"infected_population": 0.3})]
        parameters = calibrate(model, data, num_iterations=2)

        self.assertIsNotNone(parameters)

    def test_sample(self):
        '''Test the sample function.'''
        model = load_petri_model(self.filename)
        model = setup_model(model, self.initial_time, self.initial_state)
        
        timepoints = [0.2, 0.4, 0.6]
        num_samples = 10
        # Test that sample works without inferred parameters
        simulation = sample(model, timepoints, num_samples)
        
        self.assertEqual(simulation['infected_population_sol'].shape[0], num_samples)
        self.assertEqual(simulation['infected_population_sol'].shape[1], len(timepoints))
        
        data = [(0.2, {"infected_population": 0.1}), (0.4, {"infected_population": 0.2}), (0.6, {"infected_population": 0.3})]
        parameters = calibrate(model, data, num_iterations=2)
        # Test that sample works with inferred parameters
        simulation = sample(model, timepoints, num_samples, parameters)

        self.assertEqual(simulation['infected_population_sol'].shape[0], num_samples)
        self.assertEqual(simulation['infected_population_sol'].shape[1], len(timepoints))

        # Test that samples are different when num_samples > 1
        self.assertTrue(torch.all(simulation['infected_population_sol'][0, :] != simulation['infected_population_sol'][1, :]))

    def test_sample_from_mira_registry(self):
        '''Test the sample function when called on a mira.modeling.Model'''
        model = load_petri_model(sir)
        # This seems like a bit of a risky test, as mira might change...
        model = setup_model(model, self.initial_time, {"susceptible_population": 0.9, "infected_population": 0.1, "immune_population": 0.0})
        
        timepoints = [0.2, 0.4, 0.6]
        num_samples = 10
        # Test that sample works without inferred parameters
        simulation = sample(model, timepoints, num_samples)
        
        self.assertEqual(simulation['infected_population_sol'].shape[0], num_samples)
        self.assertEqual(simulation['infected_population_sol'].shape[1], len(timepoints))
        
        data = [(0.2, {"infected_population": 0.1}), (0.4, {"infected_population": 0.2}), (0.6, {"infected_population": 0.3})]
        parameters = calibrate(model, data, num_iterations=2)
        # Test that sample works with inferred parameters
        simulation = sample(model, timepoints, num_samples, parameters)

        self.assertEqual(simulation['infected_population_sol'].shape[0], num_samples)
        self.assertEqual(simulation['infected_population_sol'].shape[1], len(timepoints))


    # def test_optimize(self):
    #     '''Test the optimize function.'''
    #     model = load_petri_model(self.filename)
    #     model = setup_model(model, self.initial_time, self.initial_state)
    #     INTERVENTION= {"intervention1": [0.2, "beta"]}
    #     QOI = lambda y: scenario2dec_nday_average(y, contexts=["infected_population_sol"], ndays=3)
    #     timepoints_qoi = [0.1, 0.4, 0.6, 0.8, 0.9, 1.]
    #     ouu_policy = optimize(model,
    #                     timepoints=timepoints_qoi,
    #                     interventions=INTERVENTION,
    #                     qoi=QOI,
    #                     risk_bound=10.,
    #                     initial_guess=0.02,
    #                     bounds=[[0.],[3.]],
    #                     n_samples_ouu=1,
    #                     maxiter=0,
    #                     maxfeval=1)

    #     self.assertIsNotNone(ouu_policy)