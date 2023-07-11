import unittest
import os

from mira.examples.sir import sir_parameterized as sir

import torch

from pyciemss.PetriNetODE.interfaces import (
    load_petri_model,
    setup_model,
    reset_model,
    intervene,
    sample,
    calibrate,
    optimize,
)
from pyciemss.risk.qoi import scenario2dec_nday_average

import unittest
import os
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from pyciemss.PetriNetODE.interfaces import (
    load_and_sample_petri_model,
    load_and_calibrate_and_sample_petri_model,
)
from pyciemss.PetriNetODE.base import ScaledBetaNoisePetriNetODESystem

class Test_Samples_Format(unittest.TestCase):

    """Tests for the output of PetriNetODE.interfaces.load_*_sample_petri_net_model."""

    # Setup for the tests
    def setUp(self):
        # Should be using AMR model instead
        DEMO_PATH = "notebook/integration_demo/"

        ASKENET_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/sir_typed.json"
        self.num_samples = 2
        timepoints = [0.0, 1.0, 2.0, 3.0, 4.0]
        self.num_timepoints = len(timepoints)

        self.samples = load_and_sample_petri_model(
            ASKENET_PATH,
            self.num_samples,
            timepoints=timepoints,
        )

        data_path = os.path.join(DEMO_PATH, "data.csv")

        self.calibrated_samples = load_and_calibrate_and_sample_petri_model(
            ASKENET_PATH,
            data_path,
            self.num_samples,
            timepoints=timepoints,
            verbose=True,
            num_iterations=2,
        )
        self.interventions = [(1.1, "beta", 1.0), (2.1, "gamma", 0.1)]
        self.intervened_samples = load_and_sample_petri_model(
            ASKENET_PATH,
            self.num_samples,
            timepoints=timepoints,
            interventions=self.interventions,
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
                self.assertIn(col_name.split("_")[-1], ("param", "sol"))

    def test_samples_dtype(self):
        """Test that `samples` has the required data types"""
        for s in [self.samples, self.calibrated_samples]:
            self.assertEqual(s["timepoint_id"].dtype, np.int64)
            self.assertEqual(s["sample_id"].dtype, np.int64)
            for col_name in s.columns[2:]:
                self.assertEqual(s[col_name].dtype, np.float64)


        
class TestODEInterfaces(unittest.TestCase):
    """Tests for the ODE interfaces."""

    # Setup for the tests
    def setUp(self):
        MIRA_PATH = "test/models/evaluation_examples/scenario_1/"
        AMR_URL_TEMPLATE= "https://raw.githubusercontent.com/DARPA-ASKEM/experiments/main/thin-thread-examples/mira_v2/biomodels/{biomodel_id}/model_askenet.json"
        filename = "scenario1_sir_mira.json"
        self.filename = os.path.join(MIRA_PATH, filename)
        self.initial_time = 0.0
        self.initial_state = {
            "susceptible_population": 0.99,
            "infected_population": 0.01,
            "immune_population": 0.0,
        }
        self.interventions = [(1.1, "beta", 1.0), (2.1, "gamma", 0.1)]
        self.num_samples = 2
        self.timepoints = [0.0, 1.0, 2.0, 3.0, 4.0]
        biomodels = """BIOMD0000000249	BIOMD0000000716	BIOMD0000000949	BIOMD0000000956	BIOMD0000000960	BIOMD0000000964	BIOMD0000000971	BIOMD0000000976	BIOMD0000000979	BIOMD0000000982	BIOMD0000000988	MODEL1008060000	MODEL1805230001	MODEL2111170001 BIOMD0000000294	BIOMD0000000717	BIOMD0000000950	BIOMD0000000957	BIOMD0000000962	BIOMD0000000969	BIOMD0000000972	BIOMD0000000977	BIOMD0000000980	BIOMD0000000983	BIOMD0000000991	MODEL1008060002	MODEL1808280006
BIOMD0000000715	BIOMD0000000726	BIOMD0000000955	BIOMD0000000958	BIOMD0000000963	BIOMD0000000970	BIOMD0000000974	BIOMD0000000978	BIOMD0000000981	BIOMD0000000984	BIOMD0000001045	MODEL1805220001	MODEL1808280011""".split()
        self.biomodels_tests = {biomodel_id: dict(source=AMR_URL_TEMPLATE.format(biomodel_id=biomodel_id))
                                for biomodel_id in biomodels}

    def test_load_biomodels(self):
        """Test how if all biomodels can be loaded in the AMR format."""
        for biomodel_id, biomodel in self.biomodels_tests.items():
            try:
                model = load_petri_model(biomodel["source"], compile_rate_law_p=True)
            except Exception as e:
                print(biomodel, e)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, ScaledBetaNoisePetriNetODESystem)
                    
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
            load_petri_model(self.filename, pseudocount=2.0),
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
            (0.6, {"infected_population": 0.3}),
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

    def test_load_and_sample_petri_model(self):
        """Test the load_and_sample_petri_model function with and without interventions."""
        ASKENET_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/sir_typed.json"
        interventions=[(1e-6, "beta", 1.0), (2e-6, "gamma", 0.1)]
        timepoints = [1.0, 1.1, 1.2, 1.3]
        num_samples = 3
        initial_state = {
            "S": 0.99,
            "I": 0.01,
            "R": 0.0,
        }
        expected_intervened_samples = pd.read_csv('test/test_petrinet_ode/expected_intervened_samples.csv')
        actual_intervened_samples = load_and_sample_petri_model(ASKENET_PATH, num_samples, timepoints, interventions = interventions, start_state=initial_state)
        assert_frame_equal(expected_intervened_samples, actual_intervened_samples, check_exact=False, atol=1e-5)
        

    def test_load_and_calibrate_and_sample_petri_model(self):
        """Test the load_and_sample_petri_model function with and without interventions."""
        ASKENET_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/sir_typed.json"
        interventions=[(1e-6, "beta", 1.0), (2e-6, "gamma", 0.1)]
        timepoints = [1.0, 1.1, 1.2, 1.3]
        num_samples = 3
        initial_state = {
            "S": 0.99,
            "I": 0.01,
            "R": 0.0,
        }
        expected_intervened_samples = pd.read_csv('test/test_petrinet_ode/expected_intervened_samples.csv')
        data_path = 'test/test_petrinet_ode/data.csv'
        actual_intervened_samples = load_and_calibrate_and_sample_petri_model(ASKENET_PATH, data_path, num_samples, timepoints, interventions = interventions, start_state=initial_state, num_iterations=2)
        assert_frame_equal(expected_intervened_samples, actual_intervened_samples, check_exact=False, atol=1e-5)
        
        SCENARIO_1a_H2 = 'test/models/AMR_examples/scenario1_a.json'
        scenario1a_output = load_and_sample_petri_model(SCENARIO_1a_H2, num_samples, timepoints)
        self.assertTrue(isinstance(scenario1a_output, pd.DataFrame))

        SIDARTHE = 'test/models/AMR_examples/BIOMD0000000955_askenet.json'
        sidarthe_output = load_and_sample_petri_model(SIDARTHE, num_samples, timepoints)
        self.assertTrue(isinstance(sidarthe_output, pd.DataFrame))



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
