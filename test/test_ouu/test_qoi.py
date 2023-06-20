import unittest

import numpy as np
import torch

from pyciemss.risk.qoi import scenario2dec_nday_average


class TestQOIs(unittest.TestCase):
    """'
    Tests for quantity of interest functions.
    """

    def test_qoi_ndays_sample_average(self):
        dataCube = {"I_obs": torch.tensor([[0, 0, 2, 1], [0, 1, 2, 30], [0, 3, 2, 10]])}
        actual = scenario2dec_nday_average(dataCube, ndays=3)
        expected = np.array([1, 11, 5])
        self.assertTrue(np.array_equal(actual, expected))
