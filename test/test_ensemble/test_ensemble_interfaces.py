import unittest
import os
import copy

import torch
import pandas as pd
import numpy as np

from pyciemss.PetriNetODE.base import MiraPetriNetODESystem, ScaledBetaNoisePetriNetODESystem
from pyciemss.PetriNetODE.events import Event, StartEvent, LoggingEvent, ObservationEvent, StaticParameterInterventionEvent
from pyciemss.Ensemble.base import EnsembleSystem
import pyciemss

from pyciemss.PetriNetODE.interfaces import load_petri_model
from pyciemss.Ensemble.interfaces import load_and_sample_petri_ensemble, load_and_calibrate_and_sample_ensemble_model, setup_model, reset_model, intervene, sample, calibrate, optimize

class Test_Samples_Format(unittest.TestCase):

    """Tests for the output of PetriNetODE.interfaces.load_*_sample_petri_net_model."""

    # Setup for the tests
    def setUp(self):
        DEMO_PATH = "notebook/integration_demo/"
        ASKENET_PATH_1 = "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/sir_typed.json"
        ASKENET_PATH_2 = "test/models/AMR_examples/SIDARTHE.amr.json"

        ASKENET_PATHS = [ASKENET_PATH_1, ASKENET_PATH_2]

        def solution_mapping1(model1_solution: dict) -> dict:
            return model1_solution


        def solution_mapping2(model2_solution: dict) -> dict:
            mapped_solution = {}
            mapped_solution["Susceptible"] = (
                model2_solution["Susceptible"]
                + model2_solution["Recognized"]
                + model2_solution["Threatened"]
            )

            mapped_solution["Infected"] = (
                model2_solution["Infected"]
                + model2_solution["Ailing"]
                + model2_solution["Diagnosed"]
            )

            # Model 1 doesn't include dead people, and implicitly assumes that everyone who is infected will recover.
            mapped_solution["Recovered"] = (
                model2_solution["Healed"] + model2_solution["Extinct"]
            )

            return mapped_solution

        solution_mappings = [solution_mapping1, solution_mapping2]

        weights = [0.5, 0.5]
        self.num_samples = 2
        timepoints = [0.0, 1.0, 2.0, 3.0, 4.0]
        self.num_timepoints = len(timepoints)

        self.samples = load_and_sample_petri_ensemble(
            ASKENET_PATHS, 
            weights, 
            solution_mappings,
            self.num_samples, 
            timepoints
        )

        data_path = os.path.join(DEMO_PATH, "data.csv")

        self.calibrated_samples = load_and_calibrate_and_sample_ensemble_model(
            ASKENET_PATHS,
            data_path,
            weights,
            solution_mappings,
            self.num_samples,
            timepoints,
            total_population=1000,
            num_iterations=5
        )

    def test_samples_type(self):
        """Test that `samples` is a Pandas DataFrame"""
        for s in [self.samples, self.calibrated_samples]:
            self.assertIsInstance(s, pd.DataFrame)

    def test_samples_shape(self):
        """Test that `samples` has the correct number of rows and columns"""
        for s in [self.samples, self.calibrated_samples]:
            self.assertEqual(s.shape[0], self.num_timepoints * self.num_samples)
            self.assertGreaterEqual(s.shape[1], 2)

    def test_samples_column_names(self):
        """Test that `samples` has required column names"""
        for s in [self.samples, self.calibrated_samples]:
            self.assertEqual(list(s.columns)[:2], ["timepoint_id", "sample_id"])
            for col_name in s.columns[2:]:
                self.assertIn(col_name.split("_")[-1], ("param", "sol", "weight"))

    def test_samples_dtype(self):
        """Test that `samples` has the required data types"""
        for s in [self.samples, self.calibrated_samples]:
            self.assertEqual(s["timepoint_id"].dtype, np.int64)
            self.assertEqual(s["sample_id"].dtype, np.int64)
            for col_name in s.columns[2:]:
                self.assertEqual(s[col_name].dtype, np.float64)


class TestEnsembleInterfaces(unittest.TestCase):
    '''Tests for the Ensemble interfaces.'''
    
    # Setup for the tests
    def setUp(self):
        MIRA_PATH = "test/models/april_ensemble_demo/"

        filename1 = "BIOMD0000000955_template_model.json"
        self.filename1 = os.path.join(MIRA_PATH, filename1)
        self.model1 = load_petri_model(self.filename1, add_uncertainty=True)
        self.start_state1 = {k[0]: v.data['initial_value'] for k, v in self.model1.G.variables.items()}

        filename2 = "BIOMD0000000960_template_model.json"
        self.filename2 = os.path.join(MIRA_PATH, filename2)
        self.model2 = load_petri_model(self.filename2, add_uncertainty=True)
        self.start_state2 = {k[0]: v.data['initial_value'] for k, v in self.model2.G.variables.items()}

        self.initial_time = 0.0

        solution_ratio = self.start_state2['Infectious'] / self.start_state1['Infected']
        self.solution_mapping1 = lambda x : {"Infected": x["Infected"]}
        self.solution_mapping2 = lambda x: {"Infected": x["Infectious"] / solution_ratio}

        self.models = [self.model1, self.model2]
        self.weights = [0.5, 0.5]
        self.solution_mappings = [self.solution_mapping1, self.solution_mapping2]
        self.total_population = 1.0
        self.dirichlet_concentration = 1.0
        self.noise_pseudocount = 1.0

    def test_solution_mapping(self):
        '''Test the solution_mapping function.'''

        # Test that the solution_mapping results in the same variables.
        self.assertEqual(set(self.solution_mapping1(self.start_state1).keys()), 
                         set(self.solution_mapping2(self.start_state2).keys()))

        #Assert that the solution_mapping results in the same values.
        self.assertAlmostEqual(self.solution_mapping1(self.start_state1)["Infected"], 
                            self.solution_mapping2(self.start_state2)["Infected"])

    def test_setup_model(self):
        '''Test the setup_model function.'''
        ensemble = setup_model(self.models, self.weights, self.solution_mappings, self.initial_time, [self.start_state1, self.start_state2], self.total_population, self.noise_pseudocount, self.dirichlet_concentration)

        # Test that the model is an EnsembleSystem
        self.assertIsInstance(ensemble, EnsembleSystem)

        # Test that the model has the correct number of models
        self.assertEqual(len(ensemble.models), len(self.models))

        # Test that the model has the correct number of weights
        self.assertEqual(len(ensemble.dirichlet_alpha), len(self.weights))

        # Test that the model had the correct weight values
        self.assertTrue(torch.all(ensemble.dirichlet_alpha == torch.as_tensor(self.weights) * self.dirichlet_concentration))
        self.assertTrue(torch.all(ensemble.dirichlet_alpha == torch.tensor([0.5, 0.5])))        

        # Test that the model has the correct number of solution_mappings
        self.assertEqual(len(ensemble.solution_mappings), len(self.solution_mappings))

        # Test that models, weights, and solution_mappings have the same length
        self.assertEqual(len(ensemble.models), len(ensemble.dirichlet_alpha))
        self.assertEqual(len(ensemble.models), len(ensemble.solution_mappings))

    def test_calibrate(self):
        '''Test the calibrate function.'''
        ensemble = setup_model(self.models, self.weights, self.solution_mappings, self.initial_time, [self.start_state1, self.start_state2], self.total_population, self.noise_pseudocount, self.dirichlet_concentration)
        
        data = [(1.1, {"Infected": 0.003}), (1.2, {"Infected": 0.005})]
        parameters = calibrate(ensemble, data, num_iterations=2)

        self.assertIsNotNone(parameters)

    def test_sample(self):
        '''Test the sample function.'''
        ensemble = setup_model(self.models, self.weights, self.solution_mappings, self.initial_time, [self.start_state1, self.start_state2], self.total_population, self.noise_pseudocount, self.dirichlet_concentration)
        
        timepoints = [1.0, 5.0, 10.0]
        num_samples = 10
        # Test that sample works without inferred parameters
        simulation = sample(ensemble, timepoints, num_samples)
        
        self.assertEqual(simulation['Infected_sol'].shape[0], num_samples)
        self.assertEqual(simulation['Infected_sol'].shape[1], len(timepoints))
        
        data = [(0.2, {"Infected": 0.1}), (0.4, {"Infected": 0.2}), (0.6, {"Infected": 0.3})]
        parameters = calibrate(ensemble, data, num_iterations=2)
        # Test that sample works with inferred parameters
        simulation = sample(ensemble, timepoints, num_samples, parameters)

        self.assertEqual(simulation['Infected_sol'].shape[0], num_samples)
        self.assertEqual(simulation['Infected_sol'].shape[1], len(timepoints))

        # Test that samples are different when num_samples > 1
        self.assertTrue(torch.all(simulation['Infected_sol'][0, :] != simulation['Infected_sol'][1, :]))
