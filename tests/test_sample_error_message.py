import unittest

import torch

from pyciemss.interfaces import sample


# Unit test to assert that incorrect num_samples input to sample is caught
class TestSampleErrorMessage(unittest.TestCase):
    model_path = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/models/"
    model_location = model_path + "SEIRHD_NPI_Type1_petrinet.json"
    end_time = 100.0
    logging_step_size = 10.0

    def test_positive_int(self):
        # Assert that providing a positive int does not raise an error
        try:
            sample(
                self.model_location,
                self.end_time,
                self.logging_step_size,
                num_samples=10,
            )
        except ValueError:
            self.fail("Providing a positive integer erroneously raised a ValueError")

    def test_float(self):
        # Assert that providing a float raises a ValueError
        with self.assertRaises(ValueError):
            sample(
                self.model_location,
                self.end_time,
                self.logging_step_size,
                num_samples=10.5,
            )

    def test_negative_int(self):
        # Assert that providing a negative integer raises a ValueError
        with self.assertRaises(ValueError):
            sample(
                self.model_location,
                self.end_time,
                self.logging_step_size,
                num_samples=-10,
            )

    def test_zero(self):
        # Assert that providing 0 raises a ValueError
        with self.assertRaises(ValueError):
            sample(
                self.model_location,
                self.end_time,
                self.logging_step_size,
                num_samples=0,
            )

    def test_tensor_int(self):
        # Assert that providing a tensor with an int raises a ValueError
        with self.assertRaises(ValueError):
            sample(
                self.model_location,
                self.end_time,
                self.logging_step_size,
                num_samples=torch.tensor(10),
            )


if __name__ == "__main__":
    unittest.main()
