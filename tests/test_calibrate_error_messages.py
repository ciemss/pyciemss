import unittest

import torch

from pyciemss.interfaces import calibrate


# Unit test to assert that incorrect input to calibrate is caught
class TestCalibrateErrorMessages(unittest.TestCase):
    model_path = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/models/"
    data_path = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/datasets/"
    model_location = model_path + "SEIRHD_NPI_Type1_petrinet.json"
    data_location = data_path + "traditional.csv"
    data_mapping = {"Infected": "I"}

    def test_positive_int(self):
        # Assert that providing a positive int does not raise an error
        try:
            calibrate(
                self.model_location,
                self.data_location,
                data_mapping=self.data_mapping,
                num_iterations=3,
            )
        except ValueError:
            self.fail("Providing a positive integer erroneously raised a ValueError")

    def test_float(self):
        # Assert that providing a float raises a ValueError
        with self.assertRaises(ValueError):
            calibrate(
                self.model_location,
                self.data_location,
                data_mapping=self.data_mapping,
                num_iterations=3.5,
            )

    def test_negative_int(self):
        # Assert that providing a negative integer raises a ValueError
        with self.assertRaises(ValueError):
            calibrate(
                self.model_location,
                self.data_location,
                data_mapping=self.data_mapping,
                num_iterations=-3,
            )

    def test_zero(self):
        # Assert that providing 0 raises a ValueError
        with self.assertRaises(ValueError):
            calibrate(
                self.model_location,
                self.data_location,
                data_mapping=self.data_mapping,
                num_iterations=0,
            )

    def test_tensor_int(self):
        # Assert that providing a tensor with an int raises a ValueError
        with self.assertRaises(ValueError):
            calibrate(
                self.model_location,
                self.data_location,
                data_mapping=self.data_mapping,
                num_iterations=torch.tensor(5),
            )


if __name__ == "__main__":
    unittest.main()
