import unittest

import os
import torch

from pyro.distributions import Uniform

from pyciemss.ODE.base import PetriNetODESystem, GaussianNoisePetriNetODESystem
from pyciemss.ODE.events import Event, ObservationEvent, LoggingEvent, StartEvent, DynamicStopEvent
import pyciemss

class TestODE(unittest.TestCase):
    '''Tests for the ODE module.'''

    # Setup for the tests
    def setUp(self):
        STARTERKIT_PATH = "test/models/starter_kit_examples/"
        filename = "CHIME-SIR/model_petri.json"
        filename = os.path.join(STARTERKIT_PATH, filename)
        self.model = GaussianNoisePetriNetODESystem.from_mira(filename)

    # Clean up after tests
    def tearDown(self):
        self.model = None

    def test_from_mira(self):
        '''Test the from_mira method.'''
        STARTERKIT_PATH = "test/models/starter_kit_examples/"
        filename = "CHIME-SIR/model_petri.json"
        filename = os.path.join(STARTERKIT_PATH, filename)
        model = PetriNetODESystem.from_mira(filename)
        self.assertIsNotNone(model)

    def test_from_mira_with_noise(self):
        self.assertIsNotNone(self.model)

    def test_load_delete_start_event(self):
        '''Test the load_start_event and the delete_start_event methods.'''
        self.model.load_start_event(0.0, {"S": 0.9, "I": 0.1, "R": 0.0})
        
        self.assertEqual(self.model._start_event.time, torch.tensor(0.0))
        self.assertEqual(self.model._start_event.initial_state["S"], torch.tensor(0.9))
        self.assertEqual(self.model._start_event.initial_state["I"], torch.tensor(0.1))
        self.assertEqual(self.model._start_event.initial_state["R"], torch.tensor(0.0))
        
        self.assertEqual(len(self.model._static_events), 1)

        self.model.delete_start_event()
        self.assertEqual(self.model._start_event.initial_state, {})

    def test_load_delete_logging_event(self):
        '''Test the load_logging_event method.'''
        log_times = [1., 2.]
        self.model.load_logging_events(log_times)

        self.assertEqual(self.model._logging_events[0].time, 1.0)
        self.assertEqual(self.model._logging_events[1].time, 2.0)
        self.assertEqual(len(self.model._logging_events), 2)
        # Includes the default start event
        self.assertEqual(len(self.model._static_events), 3)

        self.model.delete_logging_events()

        self.assertEqual(len(self.model._logging_events), 0)
        # Includes the default start event
        self.assertEqual(len(self.model._static_events), 1)

    def test_load_delete_observation_events(self):
        '''Test the load_observation_events and the delete_observation_events methods.'''
        self.model.load_observation_events({0.01: {"S": 0.9, "I": 0.1}, 1.0: {"S": 0.8}})
        
        self.assertEqual(len(self.model._observation_events), 2)
        self.assertEqual(len(self.model._static_events), 3)

        self.assertEqual(self.model._observation_events[0].time, torch.tensor(0.01))
        self.assertEqual(self.model._observation_events[1].time, torch.tensor(1.0))
        self.assertEqual(self.model._observation_events[0].observation["S"], torch.tensor(0.9))
        self.assertEqual(self.model._observation_events[0].observation["I"], torch.tensor(0.1))
        self.assertEqual(self.model._observation_events[1].observation["S"], torch.tensor(0.8))

        self.assertEqual(set(self.model._observation_var_names), {"S", "I"})

        self.assertEqual(self.model._observation_indices["S"], [1, 2])
        self.assertEqual(self.model._observation_indices["I"], [1])

        self.assertTrue(torch.equal(self.model._observation_values["S"], torch.tensor([0.9, 0.8]))) 
        self.assertTrue(torch.equal(self.model._observation_values["I"], torch.tensor([0.1])))

        self.model.delete_observation_events()

        self.assertEqual(len(self.model._observation_events), 0)
        self.assertEqual(len(self.model._static_events), 1)

        self.assertEqual(self.model._observation_var_names, [])
        self.assertEqual(self.model._observation_indices, {})
        self.assertEqual(self.model._observation_values, {})
        
