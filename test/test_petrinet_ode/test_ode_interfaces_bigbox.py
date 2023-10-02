import unittest
import os
import torch
import requests
import warnings

import numpy as np
import pandas as pd


from mira.examples.sir import sir_parameterized as sir

from pyciemss.PetriNetODE.interfaces import (
    load_petri_model,
    setup_model,
    reset_model,
    intervene,
    sample,
    calibrate,
    get_posterior_density_petri,
    get_posterior_density_mesh_petri,
)

from pandas.testing import assert_frame_equal
from pyciemss.PetriNetODE.interfaces_bigbox import (
    load_and_sample_petri_model,
    load_and_calibrate_and_sample_petri_model,
    load_and_optimize_and_sample_petri_model,
    load_and_calibrate_and_optimize_and_sample_petri_model,
)

REMOTE_ASKENET_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/sir_typed.json"  # noqa
LOCAL_ASKENET_PATH = "test/test_petrinet_ode/cached_sir_typed.json"


def determine_askenet_path():
    """Some VPN configurations prevent downloading the remote schema.
    This function tries that download, but if it fails falls back to a local copy.
    Retaining the remote download helps detect issues as the cannonical schemas
    change.  BUT having a backup enables faster testing/iteration.
    """
    try:
        requests.get(REMOTE_ASKENET_PATH)
        return REMOTE_ASKENET_PATH
    except requests.exceptions.SSLError:
        warnings.warn("Using locally cached schema because remote copy is unavailable")
        return LOCAL_ASKENET_PATH


class TestSamplesFormat(unittest.TestCase):

    """Tests for the output of PetriNetODE.interfaces.load_*_sample_petri_net_model."""

    # Setup for the tests
    @classmethod
    def setUpClass(cls):
        # Should be using AMR model instead
        DEMO_PATH = "notebook/integration_demo/"

        ASKENET_PATH = determine_askenet_path()

        cls.num_samples = 2
        timepoints = [0.0, 1.0, 2.0, 3.0, 4.0]
        cls.num_timepoints = len(timepoints)
        data_path = os.path.join(DEMO_PATH, "data.csv")

        cls.samples = load_and_sample_petri_model(
            ASKENET_PATH,
            cls.num_samples,
            timepoints=timepoints,
            method="euler",
        )["data"]

        cls.calibrated_samples = load_and_calibrate_and_sample_petri_model(
            ASKENET_PATH,
            data_path,
            cls.num_samples,
            timepoints=timepoints,
            num_iterations=2,
            method="euler",
        )["data"]

        cls.calibrated_samples_deterministic = (
            load_and_calibrate_and_sample_petri_model(
                ASKENET_PATH,
                data_path,
                cls.num_samples,
                timepoints=timepoints,
                num_iterations=2,
                deterministic_learnable_parameters=["beta"],
            )["data"]
        )

        cls.interventions = [(1.0, "beta", 1.0), (2.1, "gamma", 0.1)]
        cls.intervened_samples = load_and_sample_petri_model(
            ASKENET_PATH,
            cls.num_samples,
            timepoints=timepoints,
            interventions=cls.interventions,
        )["data"]

        def OBJFUN(x):
            np.abs(x)

        INTERVENTION = [(0.1, "beta")]
        QOI = ("scenario2dec_nday_average", "I_sol", 2)

        cls.ouu_samples = load_and_optimize_and_sample_petri_model(
            ASKENET_PATH,
            cls.num_samples,
            timepoints=timepoints,
            interventions=INTERVENTION,
            qoi=QOI,
            risk_bound=10.0,
            objfun=OBJFUN,
            initial_guess=0.02,
            bounds=[[0.0], [3.0]],
            n_samples_ouu=int(1),
            maxiter=0,
            maxfeval=2,
            method="euler",
        )["data"]

        cls.ouu_cal_samples = load_and_calibrate_and_optimize_and_sample_petri_model(
            ASKENET_PATH,
            data_path,
            cls.num_samples,
            timepoints=timepoints,
            interventions=INTERVENTION,
            qoi=QOI,
            risk_bound=10.0,
            objfun=OBJFUN,
            initial_guess=0.02,
            bounds=[[0.0], [3.0]],
            num_iterations=2,
            n_samples_ouu=int(1),
            maxiter=0,
            maxfeval=2,
            method="euler",
        )["data"]

        cls.all_samples = [
            cls.samples,
            cls.calibrated_samples,
            cls.calibrated_samples_deterministic,
            cls.intervened_samples,
            cls.ouu_samples,
            cls.ouu_cal_samples,
        ]

    def test_deterministic_calibrated_samples(self):
        # Add additional test that checks that the deterministic parameters are actually deterministic
        beta = self.calibrated_samples_deterministic["beta_param"].values
        gamma = self.calibrated_samples_deterministic["gamma_param"].values
        self.assertEqual(
            beta[: self.num_timepoints].tolist(),
            beta[self.num_timepoints :].tolist(),
            "beta is deterministic during calibration",
        )
        self.assertNotEqual(
            gamma[: self.num_timepoints].tolist(),
            gamma[self.num_timepoints :].tolist(),
            "gamma is not deterministic during calibration",
        )

    def test_samples_type(self):
        """Test that `samples` is a Pandas DataFrame"""
        for s in self.all_samples:
            self.assertIsInstance(s, pd.DataFrame)

    def test_samples_shape(self):
        """Test that `samples` has the correct number of rows and columns"""
        for s in self.all_samples:
            self.assertEqual(s.shape[0], self.num_timepoints * self.num_samples)
            self.assertGreaterEqual(s.shape[1], 2)

    def test_samples_column_names(self):
        """Test that `samples` has required column names"""
        for s in self.all_samples:
            self.assertEqual(list(s.columns)[:2], ["timepoint_id", "sample_id"])
            for col_name in s.columns[2:]:
                self.assertIn(col_name.split("_")[-1], ("param", "sol", "(unknown)"))

    def test_samples_dtype(self):
        """Test that `samples` has the required data types"""
        for s in self.all_samples:
            self.assertEqual(s["timepoint_id"].dtype, np.int64)
            self.assertEqual(s["sample_id"].dtype, np.int64)
            for col_name in s.columns[2:]:
                self.assertEqual(s[col_name].dtype, np.float64)


class TestAMRDistribution(unittest.TestCase):
    """Tests for the distribution of the AMR model."""

    def test_distribution(self):
        filepath = "test/models/AMR_examples/scenario1_c_with_distributions.json"
        samples = load_and_sample_petri_model(
            filepath,
            2,
            timepoints=[1.0],
            method="euler",
        )["data"]
        self.assertIsNotNone(samples)

        k_2 = samples["k_2_param"].values
        beta_nc = samples["beta_nc_param"].values
        beta_s = samples["beta_s_param"].values

        self.assertNotEqual(
            k_2[0],
            k_2[1],
            "k_2 is drawn from a distribution and should produce different samples",
        )
        self.assertNotEqual(
            beta_nc[0],
            beta_nc[1],
            "beta_nc is drawn from a distribution and should produce different samples",
        )
        self.assertEqual(
            beta_s[0],
            beta_s[1],
            "beta_s is not drawn from a distribution and should produce the same samples",
        )


class TestProblematicCalibration(unittest.TestCase):
    """Tests for the calibration of problematic models."""

    def test_normal_noise_fix(self):
        filepath = (
            "test/test_petrinet_ode/problematic_example/BIOMD0000000955_askenet.json"
        )
        data_path = (
            "test/test_petrinet_ode/problematic_example/ciemss-dataset-input.csv"
        )
        calibrated_samples = load_and_calibrate_and_sample_petri_model(
            filepath,
            data_path,
            5,
            timepoints=[1.0, 10.0],
            num_iterations=10,
            noise_model="scaled_normal",
        )
        self.assertIsNotNone(calibrated_samples["data"])

        try:
            load_and_calibrate_and_sample_petri_model(
                filepath,
                data_path,
                5,
                timepoints=[1.0, 10.0],
                num_iterations=10,
                noise_model="scaled_beta",
            )
        except Exception as e:
            self.assertIn("Expected parameter concentration", str(e))


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
        self.interventions = [(1.0, "beta", 1.0), (2.1, "gamma", 0.1)]
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
            (1.0, {"infected_population": 0.3}),
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

        ASKENET_PATH = determine_askenet_path()

        interventions = [(1e-6, "beta", 1.0), (2e-6, "gamma", 0.1)]
        timepoints = [1.0, 1.1, 1.2, 1.3]
        num_samples = 3
        initial_state = {
            "S": 0.99,
            "I": 0.01,
            "R": 0.0,
        }
        expected_intervened_samples = pd.read_csv(
            "test/test_petrinet_ode/expected_intervened_samples.csv"
        )
        actual_intervened_samples = load_and_sample_petri_model(
            ASKENET_PATH,
            num_samples,
            timepoints,
            interventions=interventions,
            start_state=initial_state,
        )
        assert_frame_equal(
            expected_intervened_samples,
            actual_intervened_samples["data"],
            check_exact=False,
            atol=1e-5,
        )

    def test_load_and_calibrate_and_sample_petri_model(self):
        """Test the load_and_calibrate_and_sample_petri_model function with and without interventions."""
        ASKENET_PATH = determine_askenet_path()
        interventions = [(1e-6, "beta", 1.0), (2e-6, "gamma", 0.1)]
        timepoints = [1.0, 1.1, 1.2, 1.3]
        num_samples = 3
        initial_state = {
            "S": 0.99,
            "I": 0.01,
            "R": 0.0,
        }
        expected_intervened_samples = pd.read_csv(
            "test/test_petrinet_ode/expected_intervened_samples.csv"
        )
        data_path = "test/test_petrinet_ode/data.csv"
        actual_intervened_samples = load_and_calibrate_and_sample_petri_model(
            ASKENET_PATH,
            data_path,
            num_samples,
            timepoints,
            interventions=interventions,
            start_state=initial_state,
            num_iterations=2,
        )
        assert_frame_equal(
            expected_intervened_samples,
            actual_intervened_samples["data"],
            check_exact=False,
            atol=1e-5,
        )

        SCENARIO_1a_H2 = "test/models/AMR_examples/scenario1_a.json"
        scenario1a_output = load_and_sample_petri_model(
            SCENARIO_1a_H2, num_samples, timepoints
        )
        self.assertIsInstance(
            scenario1a_output["data"], pd.DataFrame, "Dataframe not returned"
        )

        SIDARTHE = "test/models/AMR_examples/BIOMD0000000955_askenet.json"
        sidarthe_output = load_and_sample_petri_model(SIDARTHE, num_samples, timepoints)
        self.assertIsInstance(
            sidarthe_output["data"], pd.DataFrame, "Dataframe not returned"
        )

    def test_get_posterior_density_mesh_petri(self):
        ASKENET_PATH = determine_askenet_path()

        timepoints = [1.0, 1.1, 1.2, 1.3]
        num_samples = 3
        initial_state = {
            "S": 0.99,
            "I": 0.01,
            "R": 0.0,
        }

        data_path = "test/test_petrinet_ode/data.csv"
        calibrated_results = load_and_calibrate_and_sample_petri_model(
            ASKENET_PATH,
            data_path,
            num_samples,
            timepoints,
            start_state=initial_state,
            num_iterations=2,
        )

        inferred_parameters = calibrated_results["inferred_parameters"]

        # Values of beta and gamma were set by looking at the priors in the model in ASKENET_PATH
        beta_params = (-0.5, 0.5, 100)
        gamma_params = (-1.0, 0.15, 50)
        betas, gammas = torch.meshgrid(
            torch.linspace(*beta_params), torch.linspace(*gamma_params), indexing="ij"
        )

        ref_density = get_posterior_density_petri(
            inferred_parameters=inferred_parameters,
            parameter_values={"beta": betas, "gamma": gammas},
        )

        params, obs_density = get_posterior_density_mesh_petri(
            inferred_parameters=inferred_parameters,
            mesh_params={"beta": beta_params, "gamma": gamma_params},
        )

        self.assertSetEqual(
            {"beta", "gamma"}, set(params.keys()), "Result keys not as expected"
        )

        for ref, obs in zip(ref_density.ravel(), obs_density.ravel()):
            self.assertAlmostEqual(ref, obs, "Result values not as expected")

    def test_get_posterior_density_petri(self):
        ASKENET_PATH = determine_askenet_path()
        timepoints = [1.0, 1.1, 1.2, 1.3]
        num_samples = 3
        initial_state = {
            "S": 0.99,
            "I": 0.01,
            "R": 0.0,
        }

        data_path = "test/test_petrinet_ode/data.csv"
        calibrated_results = load_and_calibrate_and_sample_petri_model(
            ASKENET_PATH,
            data_path,
            num_samples,
            timepoints,
            start_state=initial_state,
            num_iterations=2,
        )

        inferred_parameters = calibrated_results["inferred_parameters"]

        # Values of beta and gamma were set by looking at the priors in the model in ASKENET_PATH
        betas = torch.tensor([-1.0, 0.027])
        gammas = torch.tensor([-1.0, 0.15])

        density = get_posterior_density_petri(
            inferred_parameters=inferred_parameters,
            parameter_values={"beta": betas, "gamma": gammas},
        )

        # Density should be 0 outside of the support.
        self.assertAlmostEqual(density[0].item(), 0.0)

        # Density should be greater than 0 inside the support.
        self.assertGreater(density[1].item(), 0.0)

    # def test_optimize(self):
    #     '''Test the optimize function.'''
    #     model = load_petri_model(self.filename)
    #     model = setup_model(model, self.initial_time, self.initial_state)
    #     INTERVENTION= [(0.2, "beta")]
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
