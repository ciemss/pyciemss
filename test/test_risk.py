import unittest

from pyciemss.risk.risk_measures import mean
import numpy as np

class TestRisk(unittest.TestCase):
    ''''
    Tests for risk measures.
    '''
    def test_mean(self):
        actual = mean(np.array([0, 1, 2]))
        expected = 1
        self.assertEqual(actual, expected)
