import unittest
import os

import pandas as pd
import numpy as np

from pyciemss.Ensemble.interfaces_bigbox import (
    load_and_sample_petri_ensemble,
    load_and_calibrate_and_sample_ensemble_model,
)


class Test_Samples_Format(unittest.TestCase):
    """Tests for the output of PetriNetODE.interfaces.load_*_sample_petri_net_model."""

    # Setup for the tests
    def setUp(self):
        DEMO_PATH = "notebook/integration_demo/"
        ASKENET_PATH_1 = "test/models/AMR_examples/ensemble/SEIARHDS_AMR.json"
        ASKENET_PATH_2 = "test/models/AMR_examples/ensemble/SIRHD_AMR.json"

        ASKENET_PATHS = [ASKENET_PATH_1, ASKENET_PATH_2]

        weights = [0.5, 0.5]
        self.num_samples = 2
        timepoints = [0.0, 1.0, 2.0, 3.0, 4.0]
        self.num_timepoints = len(timepoints)
        solution_mappings = [
            {
                "Infected": "Cases",
                "Hospitalizations": "hospitalized_population",
            },  # model 1 mappings
            {
                "Infected": "Infections",
                "Hospitalizations": "hospitalized_population",
            },  # model 2 mappings
        ]

        result_ensemble = load_and_sample_petri_ensemble(
            ASKENET_PATHS, weights, solution_mappings, self.num_samples, timepoints
        )

        data_path = os.path.join(DEMO_PATH, "results_petri_ensemble/ensemble_data.csv")

        result_cal_ensemble = load_and_calibrate_and_sample_ensemble_model(
            ASKENET_PATHS,
            data_path,
            weights,
            solution_mappings,
            self.num_samples,
            timepoints,
            total_population=1000,
            num_iterations=5,
            visual_options={"title": "Calibrated Ensemble", "keep": ".*_sol"},
        )
        self.samples = result_ensemble["data"]
        self.q_ensemble = result_ensemble["quantiles"]
        self.calibrated_samples = result_cal_ensemble["data"]
        self.calibrated_q_ensemble = result_cal_ensemble["quantiles"]

    def test_samples_type(self):
        """Test that `samples` is a Pandas DataFrame"""
        for s in [
            self.samples,
            self.q_ensemble,
            self.calibrated_samples,
            self.calibrated_q_ensemble,
        ]:
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
                self.assertIn(
                    col_name.split("_")[-1], ("param", "sol", "weight", "(unknown)")
                )

    def test_samples_dtype(self):
        """Test that `samples` has the required data types"""
        for s in [self.samples, self.calibrated_samples]:
            self.assertEqual(s["timepoint_id"].dtype, np.int64)
            self.assertEqual(s["sample_id"].dtype, np.int64)
            for col_name in s.columns[2:]:
                self.assertEqual(s[col_name].dtype, np.float64)
