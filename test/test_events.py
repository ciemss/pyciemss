import unittest

import torch
from pyciemss.ODE.events import (
    Event,
    StaticEvent,
    ObservationEvent,
    LoggingEvent,
    StartEvent,
    StaticParameterInterventionEvent,
)


class TestEvents(unittest.TestCase):
    """Tests for the events module."""

    def test_event(self):
        """Test the Event base class."""
        event = Event()
        self.assertIsNotNone(event)

    def test_static_event(self):
        """Test the StaticEvent class."""
        time = 1.0
        static_event = StaticEvent(time)
        self.assertIsNotNone(static_event)
        self.assertEqual(static_event.time, time)

    def test_observation_event(self):
        """Test the ObservationEvent class."""
        time = 1.0
        observation = {"a": 1.0}
        observation_event = ObservationEvent(time, observation)
        self.assertIsNotNone(observation_event)
        self.assertEqual(observation_event.time, time)
        self.assertEqual(observation_event.observation, observation)

    def test_start_event(self):
        """Test the StartEvent class."""
        time = 1.0
        initial_state = {"s": 1.0}
        start_event = StartEvent(time, initial_state)
        self.assertIsNotNone(start_event)
        self.assertEqual(start_event.time, time)

    def test_logging_event(self):
        """Test the LoggingEvent class."""
        time = 1.0
        logging_event = LoggingEvent(time)
        self.assertIsNotNone(logging_event)
        self.assertEqual(logging_event.time, time)

    def test_static_parameter_intervention_event(self):
        """Test the StaticParameterInterventionEvent class."""
        time = 1.0
        parameter = "a"
        value = 1.0
        static_parameter_intervention_event = StaticParameterInterventionEvent(
            time, parameter, value
        )
        self.assertIsNotNone(static_parameter_intervention_event)
        self.assertEqual(static_parameter_intervention_event.time, time)
        self.assertEqual(static_parameter_intervention_event.parameter, parameter)
        self.assertEqual(static_parameter_intervention_event.value, value)

    def test_lt(self):
        """Test the __lt__ method."""
        time1 = 1.0
        time2 = 2.0
        static_event1 = StaticEvent(time1)
        static_event2 = StaticEvent(time2)
        self.assertTrue(static_event1 < static_event2)
        self.assertFalse(static_event2 < static_event1)
