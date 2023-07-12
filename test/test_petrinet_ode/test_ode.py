import unittest

import os
from copy import deepcopy
import torch
import pyciemss
from pyciemss.PetriNetODE.interfaces import load_petri_model
from pyciemss.PetriNetODE.base import (
    PetriNetODESystem,
    MiraPetriNetODESystem,
    ScaledBetaNoisePetriNetODESystem,
)
from pyciemss.PetriNetODE.events import (
    ObservationEvent,
    LoggingEvent,
    StartEvent,
    StaticParameterInterventionEvent,
)
from pyciemss.PetriNetODE.models import MiraRegNetODESystem, LotkaVolterra
from pyro.infer.autoguide import AutoNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam
import pyro
from mira.sources.askenet import model_from_json_file, model_from_url
from mira.examples.sir import sir as mira_sir
import json
import mira


class TestODE(unittest.TestCase):
    """Tests for the ODE module."""

    # Setup for the tests
    def setUp(self):
        MIRA_PATH = "test/models/evaluation_examples/scenario_1/"
        filename = "scenario1_sir_mira.json"
        filename = os.path.join(MIRA_PATH, filename)
        self.model = ScaledBetaNoisePetriNetODESystem.from_mira(filename)

    # Clean up after tests
    def tearDown(self):
        self.model = None

    def test_ODE_base_class(self):
        """Test the ODE base class."""
        # Assert that the constructor requires var_order
        with self.assertRaises(NotImplementedError):
            model = PetriNetODESystem()

        # TODO: add more tests here as handwritten implementations of ODEs are added

    def test_from_mira(self):
        """Test the from_mira method."""
        MIRA_PATH = "test/models/evaluation_examples/scenario_1/"
        filename = "scenario1_sir_mira.json"
        filename = os.path.join(MIRA_PATH, filename)
        model = MiraPetriNetODESystem.from_mira(filename)
        self.assertIsNotNone(model)

    def test_from_mira_with_noise(self):
        self.assertIsNotNone(self.model)

    def test_from_askenet_petrinet(self):
        """Test the import from askenet json"""
        ASKENET_PATH = "test/models/"
        filename = "askenet_sir.json"
        filename = os.path.join(ASKENET_PATH, filename)
        mira_model = model_from_json_file(filename)
        model = MiraPetriNetODESystem.from_mira(mira_model)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, MiraPetriNetODESystem)

    def test_from_askenet_regnet(self):
        """Test the import from askenet json"""
        ASKENET_PATH = "test/models/may-hackathon"
        filename = "lotka_volterra.json"
        filename = os.path.join(ASKENET_PATH, filename)
        regnet_model = model_from_json_file(filename)
        model = MiraRegNetODESystem.from_mira(regnet_model)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, MiraRegNetODESystem)

    def test_load_remove_start_event(self):
        """Test the load_event method for StartEvent and the remove_start_event methods."""
        event = StartEvent(
            0.0,
            {
                "susceptible_population": 0.9,
                "infected_population": 0.1,
                "immune_population": 0.0,
            },
        )

        self.model.load_event(event)

        self.assertEqual(len(self.model._static_events), 1)

        self.assertEqual(self.model._static_events[0].time, torch.tensor(0.0))
        self.assertEqual(
            self.model._static_events[0].initial_state["susceptible_population"],
            torch.tensor(0.9),
        )
        self.assertEqual(
            self.model._static_events[0].initial_state["infected_population"],
            torch.tensor(0.1),
        )
        self.assertEqual(
            self.model._static_events[0].initial_state["immune_population"],
            torch.tensor(0.0),
        )

        self.model.remove_start_event()

        self.assertEqual(len(self.model._static_events), 0)

    def test_load_remove_logging_event(self):
        """Test the load_events method for LoggingEvent and the remove_logging_events methods."""
        self.model.load_events([LoggingEvent(1.0), LoggingEvent(2.0)])

        self.assertEqual(len(self.model._static_events), 2)
        self.assertEqual(self.model._static_events[0].time, 1.0)
        self.assertEqual(self.model._static_events[1].time, 2.0)

        self.model.remove_logging_events()

        self.assertEqual(len(self.model._static_events), 0)

    def test_load_remove_observation_events(self):
        """Test the load_observation_events and the remove_observation_events methods."""
        observation1 = ObservationEvent(
            0.01, {"susceptible_population": 0.9, "infected_population": 0.1}
        )
        observation2 = ObservationEvent(1.0, {"susceptible_population": 0.8})

        self.model.load_events([observation1, observation2])

        self.assertEqual(len(self.model._static_events), 2)

        self.assertEqual(self.model._static_events[0].time, torch.tensor(0.01))
        self.assertEqual(self.model._static_events[1].time, torch.tensor(1.0))
        self.assertEqual(
            self.model._static_events[0].observation["susceptible_population"],
            torch.tensor(0.9),
        )
        self.assertEqual(
            self.model._static_events[0].observation["infected_population"],
            torch.tensor(0.1),
        )
        self.assertEqual(
            self.model._static_events[1].observation["susceptible_population"],
            torch.tensor(0.8),
        )

        self.assertEqual(
            set(self.model._observation_var_names),
            {"susceptible_population", "infected_population"},
        )

        self.model.remove_observation_events()

        self.assertEqual(len(self.model._static_events), 0)

        self.assertEqual(self.model._observation_var_names, [])

    def test_load_remove_static_parameter_intervention_events(self):
        """Test the load_events method for StaticParameterIntervention and the remove_static_parameter_interventions methods."""
        # Load some static parameter intervention events
        intervention1 = StaticParameterInterventionEvent(2.99, "beta", 0.0)
        intervention2 = StaticParameterInterventionEvent(4.11, "beta", 10.0)
        self.model.load_events([intervention1, intervention2])

        self.assertEqual(len(self.model._static_events), 2)

        self.assertEqual(self.model._static_events[0].time, torch.tensor(2.99))
        self.assertEqual(self.model._static_events[1].time, torch.tensor(4.11))
        self.assertEqual(self.model._static_events[0].parameter, "beta")
        self.assertEqual(self.model._static_events[1].parameter, "beta")
        self.assertEqual(self.model._static_events[0].value, torch.tensor(0.0))
        self.assertEqual(self.model._static_events[1].value, torch.tensor(10.0))

        self.model.remove_static_parameter_intervention_events()

        self.assertEqual(len(self.model._static_events), 0)

    def test_observation_indices_and_values(self):
        """Test the _setup_observation_indices_and_values method."""

        observation1 = ObservationEvent(
            0.01, {"susceptible_population": 0.9, "infected_population": 0.1}
        )
        observation2 = ObservationEvent(1.0, {"susceptible_population": 0.8})

        self.model.load_events([observation1, observation2])

        self.assertListEqual(
            self.model._observation_var_names,
            ["susceptible_population", "infected_population"],
        )

        self.model._setup_observation_indices_and_values()

        self.assertEqual(
            self.model._observation_indices["susceptible_population"], [0, 1]
        )
        self.assertEqual(self.model._observation_indices["infected_population"], [0])

        self.assertTrue(
            torch.equal(
                self.model._observation_values["susceptible_population"],
                torch.tensor([0.9, 0.8]),
            )
        )
        self.assertTrue(
            torch.equal(
                self.model._observation_values["infected_population"],
                torch.tensor([0.1]),
            )
        )

    def test_integration(self):
        model = self.model

        # Load the start event
        start_event = StartEvent(
            0.0,
            {
                "susceptible_population": 0.9,
                "infected_population": 0.1,
                "immune_population": 0.0,
            },
        )
        model.load_event(start_event)

        # Load the logging events
        tspan = range(1, 10)
        logging_events = [LoggingEvent(t) for t in tspan]
        model.load_events(logging_events)

        # Run the model without observations
        solution = model()

        self.assertEqual(
            len(solution["infected_population"]), len(solution["immune_population"])
        )
        self.assertEqual(
            len(solution["infected_population"]),
            len(solution["susceptible_population"]),
        )
        self.assertEqual(len(solution["infected_population"]), len(tspan))

        # Susceptible individuals should decrease over time
        self.assertTrue(
            torch.all(
                solution["susceptible_population"][:-1]
                > solution["susceptible_population"][1:]
            )
        )

        # Recovered individuals should increase over time
        self.assertTrue(
            torch.all(
                solution["immune_population"][:-1] < solution["immune_population"][1:]
            )
        )

        # Remove the logs
        model.remove_logging_events()

        # Load the observation events
        observation1 = ObservationEvent(
            0.01, {"susceptible_population": 0.9, "infected_population": 0.1}
        )
        observation2 = ObservationEvent(1.0, {"susceptible_population": 0.8})

        self.model.load_events([observation1, observation2])

        solution = model()

        # No logging events, so we don't return anything.
        self.assertEqual(
            len(solution["infected_population"]), len(solution["immune_population"])
        )
        self.assertEqual(
            len(solution["infected_population"]),
            len(solution["susceptible_population"]),
        )
        self.assertEqual(len(solution["infected_population"]), 0)

        # Run inference
        guide = AutoNormal(model)

        optim = Adam({"lr": 0.03})
        loss_f = Trace_ELBO(num_particles=1)

        svi = SVI(model, guide, optim, loss=loss_f)

        pyro.clear_param_store()

        # Step once to setup parameters
        svi.step()
        old_params = deepcopy(list(guide.parameters()))

        # Check that the parameters have been set
        self.assertEqual(len(old_params), 4)

        # Step again to update parameters
        svi.step()

        # Check that the parameters have been updated
        for i, p in enumerate(guide.parameters()):
            self.assertNotEqual(p, old_params[i])

        # Remove the observation events and add logging events.
        model.remove_observation_events()
        model.load_events(logging_events)

        # Add a few static parameter interventions
        intervention1 = StaticParameterInterventionEvent(2.99, "beta", 0.0)
        intervention2 = StaticParameterInterventionEvent(4.11, "beta", 10.0)
        model.load_events([intervention1, intervention2])

        # Sample from the posterior predictive distribution
        predictions = Predictive(model, guide=guide, num_samples=2)()

        self.assertEqual(
            predictions["infected_population_sol"].shape,
            predictions["immune_population_sol"].shape,
        )
        self.assertEqual(
            predictions["infected_population_sol"].shape,
            predictions["susceptible_population_sol"].shape,
        )
        self.assertEqual(
            predictions["infected_population_sol"].shape, torch.Size([2, 9])
        )

        # Susceptible individuals shouldn't change between t=3 and t=4 because of the first intervention
        self.assertTrue(
            torch.all(
                torch.isclose(
                    predictions["susceptible_population_sol"][:, 2],
                    predictions["susceptible_population_sol"][:, 3],
                )
            )
        )

        # Recovered individuals should increase between t=3 and t=4
        self.assertTrue(
            torch.all(
                predictions["immune_population_sol"][:, 2]
                < predictions["immune_population_sol"][:, 3]
            )
        )

        # Susceptible individuals should decrease between t=4 and t=5 because of the second intervention
        self.assertTrue(
            torch.all(
                predictions["susceptible_population_sol"][:, 3]
                > predictions["susceptible_population_sol"][:, 4]
            )
        )
