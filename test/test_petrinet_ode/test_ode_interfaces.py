import unittest
import os
import pyro
import torch

from mira.examples.sir import sir_parameterized as sir
from pyro.infer.autoguide import AutoDelta, AutoLowRankMultivariateNormal, AutoGuideList


from pyciemss.PetriNetODE.interfaces import (
    load_petri_model,
    setup_model,
    reset_model,
    intervene,
    sample,
    calibrate,
)

from pyciemss.utils.interface_utils import csv_to_list

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from pyciemss.PetriNetODE.interfaces import (
    get_posterior_density_petri,
    get_posterior_density_mesh_petri
)



class TestODEInterfaces(unittest.TestCase):
    """Tests for the ODE interfaces."""

    # Setup for the tests
    def setUp(self):
        MIRA_PATH = "test/models/evaluation_examples/scenario_1/"

        filename = "scenario1_sir_mira.json"
        self.filename = os.path.join(MIRA_PATH, filename)
        self.initial_time = 0.0
        self.initial_state = {
            "susceptible_population": 0.99,
            "infected_population": 0.01,
            "immune_population": 0.0,
        }
        self.interventions = [(1., "beta", 1.0), (2.1, "gamma", 0.1)]
        self.num_samples = 2
        self.timepoints = [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_load_petri_from_file(self):
        """Test the load_petri function when called on a string."""
        model = load_petri_model(self.filename, add_uncertainty=True)
        self.assertIsNotNone(model)

        model = load_petri_model(self.filename, add_uncertainty=False)
        self.assertIsNotNone(model)

    def test_load_petri_from_mira_registry(self):
        """Test the load_petri function when called on a mira.modeling.Model"""
        model = load_petri_model(sir, add_uncertainty=True)
        self.assertIsNotNone(model)

        model = load_petri_model(sir, add_uncertainty=False)
        self.assertIsNotNone(model)

    def test_setup_model(self):
        """Test the setup_model function."""
        for model in [
            load_petri_model(self.filename),
            load_petri_model(self.filename, noise_scale=0.5),
        ]:
            new_model = setup_model(model, self.initial_time, self.initial_state)

            self.assertIsNotNone(new_model)
            self.assertEqual(len(new_model._static_events), 1)

            # Check that setup_model is not inplace.
            self.assertEqual(len(model._static_events), 0)

    def test_reset_model(self):
        """Test the reset_model function."""

        model = load_petri_model(self.filename)
        model = setup_model(model, self.initial_time, self.initial_state)
        self.assertEqual(len(model._static_events), 1)

        new_model = reset_model(model)
        self.assertEqual(len(new_model._static_events), 0)

        # Check that reset_model is not inplace.
        self.assertEqual(len(model._static_events), 1)

    def test_intervene(self):
        """Test the intervene function."""
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
        """Test the calibrate function."""
        model = load_petri_model(self.filename)
        model = setup_model(model, self.initial_time, self.initial_state)

        data = [
            (0.2, {"infected_population": 0.1}),
            (0.4, {"infected_population": 0.2}),
            (1., {"infected_population": 0.3}),
        ]
        parameters = calibrate(model, data, num_iterations=2)

        self.assertIsNotNone(parameters)

    def test_sample(self):
        """Test the sample function."""
        model = load_petri_model(self.filename)
        model = setup_model(model, self.initial_time, self.initial_state)

        timepoints = [0.2, 0.4, 0.6]
        num_samples = 10
        # Test that sample works without inferred parameters
        simulation = sample(model, timepoints, num_samples)

        self.assertEqual(simulation["infected_population_sol"].shape[0], num_samples)
        self.assertEqual(
            simulation["infected_population_sol"].shape[1], len(timepoints)
        )

        data = [
            (0.2, {"infected_population": 0.1}),
            (0.4, {"infected_population": 0.2}),
            (0.6, {"infected_population": 0.3}),
        ]
        parameters = calibrate(model, data, num_iterations=2)
        # Test that sample works with inferred parameters
        simulation = sample(model, timepoints, num_samples, parameters)

        self.assertEqual(simulation["infected_population_sol"].shape[0], num_samples)
        self.assertEqual(
            simulation["infected_population_sol"].shape[1], len(timepoints)
        )

        # Test that samples are different when num_samples > 1
        self.assertTrue(
            torch.all(
                simulation["infected_population_sol"][0, :]
                != simulation["infected_population_sol"][1, :]
            )
        )

    def test_sample_from_mira_registry(self):
        """Test the sample function when called on a mira.modeling.Model"""
        model = load_petri_model(sir)
        # This seems like a bit of a risky test, as mira might change...
        model = setup_model(
            model,
            self.initial_time,
            {
                "susceptible_population": 0.9,
                "infected_population": 0.1,
                "immune_population": 0.0,
            },
        )

        timepoints = [0.2, 0.4, 0.6]
        num_samples = 10
        # Test that sample works without inferred parameters
        simulation = sample(model, timepoints, num_samples)

        self.assertEqual(simulation["infected_population_sol"].shape[0], num_samples)
        self.assertEqual(
            simulation["infected_population_sol"].shape[1], len(timepoints)
        )

        data = [
            (0.2, {"infected_population": 0.1}),
            (0.4, {"infected_population": 0.2}),
            (0.6, {"infected_population": 0.3}),
        ]
        parameters = calibrate(model, data, num_iterations=2)
        # Test that sample works with inferred parameters
        simulation = sample(model, timepoints, num_samples, parameters)

        self.assertEqual(simulation["infected_population_sol"].shape[0], num_samples)
        self.assertEqual(
            simulation["infected_population_sol"].shape[1], len(timepoints)
        )

    def test_get_posterior_density_mesh_petri(self):
        ASKENET_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/sir_typed.json"
        timepoints = [1.0, 1.1, 1.2, 1.3]
        num_samples = 3
        initial_state = {
            "S": 0.99,
            "I": 0.01,
            "R": 0.0,
        }

        def autoguide(model):
            guide = AutoGuideList(model)
            guide.append(
                AutoDelta(
                    pyro.poutine.block(model, expose=[])
                )
            )
            guide.append(
                AutoLowRankMultivariateNormal(
                    pyro.poutine.block(model, hide=[])
                )
            )
            return guide


        data_path = "test/test_petrinet_ode/data.csv"
          
        model = load_petri_model(ASKENET_PATH)
        model = setup_model(model, self.initial_time, initial_state)
        inferred_parameters = calibrate(model, csv_to_list(data_path), num_iterations=2, autoguide=autoguide)
        calibrated_results = sample(model, timepoints, num_samples, inferred_parameters)

        # Values of beta and gamma were set by looking at the priors in the model in ASKENET_PATH
        beta_params = (-.5, .5, 100)
        gamma_params = (-1., 0.15, 50)
        betas, gammas = torch.meshgrid(torch.linspace(*beta_params),
                                       torch.linspace(*gamma_params), indexing='ij')
        
        ref_density = get_posterior_density_petri(
            inferred_parameters=inferred_parameters,
            parameter_values={"beta": betas, "gamma": gammas}
        )

        params, obs_density = get_posterior_density_mesh_petri(
            inferred_parameters=inferred_parameters,
            mesh_params={"beta": beta_params, "gamma": gamma_params})

        self.assertSetEqual({"beta", "gamma"},
                            set(params.keys()),
                            "Result keys not as expected")

        for ref, obs in zip(ref_density.ravel(), obs_density.ravel()):
            self.assertAlmostEqual(ref, obs, "Result values not as expected")
      
    def test_get_posterior_density_petri(self):
        ASKENET_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/sir_typed.json"
        timepoints = [1.0, 1.1, 1.2, 1.3]
        num_samples = 3
        initial_state = {
            "S": 0.99,
            "I": 0.01,
            "R": 0.0,
        }

        def autoguide(model):
            guide = AutoGuideList(model)
            guide.append(
                AutoDelta(
                    pyro.poutine.block(model, expose=[])
                )
            )
            guide.append(
                AutoLowRankMultivariateNormal(
                    pyro.poutine.block(model, hide=[])
                )
            )
            return guide


        data_path = "test/test_petrinet_ode/data.csv"

        model = load_petri_model(ASKENET_PATH)
        model = setup_model(model,  -1e-10, initial_state)
        inferred_parameters = calibrate(model,
                                        csv_to_list(data_path),
                                        num_iterations=2,
                                        autoguide=autoguide)
        calibrated_results = sample(model, timepoints, num_samples, inferred_parameters)
        
        # Values of beta and gamma were set by looking at the priors in the model in ASKENET_PATH
        betas = torch.tensor([-1., 0.027])
        gammas = torch.tensor([-1., 0.15])
    
        density = get_posterior_density_petri(inferred_parameters=inferred_parameters, parameter_values={"beta": betas, "gamma": gammas})
        
        # Density should be 0 outside of the support.
        self.assertAlmostEqual(density[0].item(), 0.)

        # Density should be greater than 0 inside the support.
        self.assertGreater(density[1].item(), 0.)
