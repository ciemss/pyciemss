import unittest
import os


from pyciemss.PetriNetODE.events import ObservationEvent, LoggingEvent, StartEvent, StaticParameterInterventionEvent
from pyciemss.PetriNetODE.interfaces import load_petri_model, setup_model, reset_model, intervene, sample, calibrate, optimize

class TestODEInterfaces(unittest.TestCase):
    '''Tests for the ODE interfaces.'''
    
    # Setup for the tests
    def setUp(self):
        STARTERKIT_PATH = "test/models/starter_kit_examples/"
        filename = "CHIME-SIR/model_petri.json"
        self.filename = os.path.join(STARTERKIT_PATH, filename)
        self.start_event = StartEvent(0.0, {"S": 0.9, "I": 0.1, "R": 0.0})

    def test_load_petri(self):
        '''Test the load_petri function.'''
        model = load_petri_model(self.filename, add_uncertainty=True)
        self.assertIsNotNone(model)

        model = load_petri_model(self.filename, add_uncertainty=False)
        self.assertIsNotNone(model)
    
    def test_setup_model(self):
        '''Test the setup_model function.'''
        model = load_petri_model(self.filename)

        new_model = setup_model(model, self.start_event)
        
        self.assertIsNotNone(new_model)
        self.assertEqual(len(new_model._static_events), 1)
        
        # Check that setup_model is not inplace.
        self.assertEqual(len(model._static_events), 0)
        
    def test_reset_model(self):
        '''Test the reset_model function.'''
        
        model = load_petri_model(self.filename)
        model = setup_model(model, self.start_event)
        self.assertEqual(len(model._static_events), 1)
        
        new_model = reset_model(model)
        self.assertEqual(len(new_model._static_events), 0)

        # Check that reset_model is not inplace.
        self.assertEqual(len(model._static_events), 1)

    def test_intervene(self):
        '''Test the intervene function.'''
        model = load_petri_model(self.filename)
        model = setup_model(model, self.start_event)
        
        intervention_event = StaticParameterInterventionEvent(0.2, "beta", 0.1)
        new_model = intervene(model, [intervention_event])
        
        self.assertEqual(len(new_model._static_events), 2)

        # Check that intervene is not inplace.
        self.assertEqual(len(model._static_events), 1)

    def test_calibrate(self):
        '''Test the calibrate function.'''
        model = load_petri_model(self.filename)
        model = setup_model(model, self.start_event)
        
        observation_event = ObservationEvent(0.2, {"I": 0.1})
        parameters = calibrate(model, [observation_event], num_iterations=2)

        self.assertIsNotNone(parameters)

    def test_sample(self):
        '''Test the sample function.'''
        model = load_petri_model(self.filename)
        model = setup_model(model, self.start_event)
        
        timepoints = [0.2, 0.4, 0.6]
        num_samples = 10
        # Test that sample works without inferred parameters
        simulation = sample(model, timepoints, num_samples)
        
        self.assertEqual(simulation['I_sol'].shape[0], num_samples)
        self.assertEqual(simulation['I_sol'].shape[1], len(timepoints))
        

        observation_event = ObservationEvent(0.2, {"I": 0.1})
        parameters = calibrate(model, [observation_event], num_iterations=2)
        # Test that sample works with inferred parameters
        simulation = sample(model, timepoints, num_samples, parameters)

        self.assertEqual(simulation['I_sol'].shape[0], num_samples)
        self.assertEqual(simulation['I_sol'].shape[1], len(timepoints))

