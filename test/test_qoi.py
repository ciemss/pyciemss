import unittest

from pyciemss.risk.qoi import nday_rolling_average, threshold_exceedence
import numpy as np

class TestQOIs(unittest.TestCase):
    ''''
    Tests for quantity of interest functions.
    '''
    def test_qoi_nday_rolling_average(self):
        dataCube = np.array([[0, 0, 2, 1], [0, 1, 2, 30], [0, 3, 2, 10]])
        actual = nday_rolling_average(dataCube, ndays=3, tf=4., dt=1.)
        expected = np.array([1, 11, 5])
        self.assertTrue(np.array_equal(actual, expected))

    
    def test_qoi_threshold_exceedence(self):
        dataCube = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        actual = threshold_exceedence(dataCube, threshold=5)
        expected = np.array([0, 1, 1])
        self.assertTrue(np.array_equal(actual, expected))
