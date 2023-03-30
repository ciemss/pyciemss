import unittest
from pyciemss.ODE.models import SIDARTHE


class SIDARTHETest(unittest.TestCase):
    def test_initialize(self):
        self.SIDARTHE = SIDARTHE()
        self.assertIsNotNone(self.SIDARTHE)
