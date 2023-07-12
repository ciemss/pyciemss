import unittest
from pyciemss.utils.interface_utils import (
    convert_to_output_format,
    interventions_and_sampled_params_to_interval,
    assign_interventions_to_timepoints,
)
from torch import tensor
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal


class Test_Interface_Utils(unittest.TestCase):
    """Test interface_utils.py"""

    def setUp(self):
        """Set up test fixtures."""
        self.intervened_samples = {
            "beta": tensor([0.0263, 0.0271, 0.0246]),
            "gamma": tensor([0.1308, 0.1342, 0.1359]),
            "Infected_sol": tensor(
                [
                    [0.9491, 0.9008, 3.5336, 22.6421, 132.3466],
                    [0.9478, 0.8984, 3.5195, 22.5536, 131.8928],
                    [0.9459, 0.8946, 3.5016, 22.4412, 131.3172],
                ]
            ),
            "Recovered_sol": tensor(
                [
                    [0.0637, 0.1242, 0.5759, 1.6780, 8.0082],
                    [0.0653, 0.1273, 0.5787, 1.6764, 7.9834],
                    [0.0661, 0.1286, 0.5784, 1.6706, 7.9482],
                ]
            ),
            "Susceptible_sol": tensor(
                [
                    [999.9871, 999.9746, 996.8905, 976.6799, 860.6453],
                    [999.9868, 999.9745, 996.9033, 976.7705, 861.1242],
                    [999.9880, 999.9767, 996.9209, 976.8878, 861.7342],
                ]
            ),
        }
        self.interventions = [
            (1.1, "beta", 1.0),
            (2.1, "gamma", 0.1),
            (1.3, "beta", 2.0),
            (1.4, "gamma", 0.3),
        ]
        self.timepoints = [0.5, 1.0, 2.0, 3.0, 4.0]
        self.sampled_params = {
            "beta_param": [0.2, 0.25, 0.15],
            "gamma_param": [0.3, 0.25, 0.35],
        }

    def test_convert_to_output_format(self):
        """Test convert_to_output_format."""
        expected_output = pd.read_csv("test/test_utils/expected_output_format.csv")
        result = convert_to_output_format(
            self.intervened_samples,
            self.timepoints,
            self.interventions,
            time_unit="FancyUnit",
        )

        self.assertTrue(
            "timepoint_FancyUnit" in result.columns, "Unit supplied & labled"
        )
        assert_frame_equal(
            expected_output,
            result.drop(columns=["timepoint_FancyUnit"]),
            check_exact=False,
            atol=1e-5,
            rtol=1e-5,
        )

        result = convert_to_output_format(
            self.intervened_samples, self.timepoints, self.interventions
        )
        self.assertTrue("timepoint_(unknown)" in result.columns, "No unit supplied")

        result = convert_to_output_format(
            self.intervened_samples, self.timepoints, self.interventions, time_unit=None
        )
        self.assertEqual(
            1,
            len([c for c in result.columns if c.startswith("timepoint_")]),
            "Unit not requested",
        )

        assert_frame_equal(
            expected_output,
            result,
            check_exact=False,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_intervention_to_interval(self):
        """Test intervention_to_interval."""
        expected_intervals = {
            "beta_param": [
                {"start": -np.inf, "end": 1.1, "param_values": [0.2, 0.25, 0.15]},
                {"start": 1.1, "end": 1.3, "param_values": [1.0, 1.0, 1.0]},
                {"start": 1.3, "end": np.inf, "param_values": [2.0, 2.0, 2.0]},
            ],
            "gamma_param": [
                {"start": -np.inf, "end": 1.4, "param_values": [0.3, 0.25, 0.35]},
                {"start": 1.4, "end": 2.1, "param_values": [0.3, 0.3, 0.3]},
                {"start": 2.1, "end": np.inf, "param_values": [0.1, 0.1, 0.1]},
            ],
        }

        self.assertEqual(
            expected_intervals,
            interventions_and_sampled_params_to_interval(
                self.interventions, self.sampled_params
            ),
        )

    def test_assign_interventions_to_timepoints(self):
        """Test assign_interventions_to_timepoints."""
        expected_intervened_values = {
            "beta_param": [
                0.2,
                0.2,
                2.0,
                2.0,
                2.0,
                0.25,
                0.25,
                2.0,
                2.0,
                2.0,
                0.15,
                0.15,
                2.0,
                2.0,
                2.0,
            ],
            "gamma_param": [
                0.3,
                0.3,
                0.3,
                0.1,
                0.1,
                0.25,
                0.25,
                0.3,
                0.1,
                0.1,
                0.35,
                0.35,
                0.3,
                0.1,
                0.1,
            ],
        }
        self.assertEqual(
            expected_intervened_values,
            assign_interventions_to_timepoints(
                self.interventions, self.timepoints, self.sampled_params
            ),
        )
