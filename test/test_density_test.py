from pyciemss.utils import is_density_equal, is_intervention_density_equal
import pyro
import torch
import unittest
from pyro.infer import  Trace_ELBO

class TestDensityTest(unittest.TestCase):
    """Tests for comparing the density of two models."""

    def test_is_density_equal(self):
        """Tests for comparing the density of two models."""
        # Define the first model
        def model1():
            x = pyro.sample("x", pyro.distributions.Normal(0, 1))
            y = pyro.sample("y", pyro.distributions.Normal(x, 1))
            return y

        # Define the second model
        def model2():
            x = pyro.sample("x", pyro.distributions.Normal(0, 1))
            y = pyro.sample("y", pyro.distributions.Normal(x, 2))
            return y

        def model3():
            x = pyro.sample("x", pyro.distributions.Normal(0, 1))
            y = pyro.sample("y", pyro.distributions.Normal(x, 2))
            return y

        def model4():
            x = pyro.sample("x", pyro.distributions.Normal(0, 1))
            y = pyro.sample("y", pyro.distributions.Normal(-x, 2))
            return y



        # def model5():
        #     x = pyro.sample("x", pyro.distributions.Normal(0, 1))
        #     y = pyro.deterministic("y", x)
        #     z = pyro.sample("z", pyro.distributions.Normal(x - y, 1))
        #     return z

        def model6():
            x = pyro.sample("x", pyro.distributions.Normal(0, 1))
            y = pyro.sample("y", pyro.distributions.Normal(x, 1))
            z = pyro.sample("z", pyro.distributions.Normal(0, 1))

        def model7():
            x = pyro.sample("x", pyro.distributions.Normal(0, 1))
            y = pyro.sample("y", pyro.distributions.Normal(x, 1))
            return y

        def model8():
            x = pyro.sample("x", pyro.distributions.Normal(0, 1))
            y = pyro.sample("y", pyro.distributions.Normal(0, torch.sqrt(torch.tensor([2]))))
            return y

        def model9():
            y = pyro.sample("y", pyro.distributions.Normal(0, torch.sqrt(torch.tensor([2]))))
            x = pyro.sample("x", pyro.distributions.Normal(y/2, torch.sqrt(torch.tensor([2]))/2))
            return x, y

        def model10():
            x = pyro.sample("x", pyro.distributions.Normal(0, 1))
            y = pyro.sample("y", pyro.distributions.Normal(x, 1))
            return x, y

        # Define the number of samples
        num_samples = 100
        # Define the loss function
        elbo = Trace_ELBO(num_particles=num_samples, vectorize_particles=False)

        # Sample from both models
        self.assertAlmostEqual( elbo.loss(model1, model1),elbo.loss(model1, model1))
        self.assertNotEqual(elbo.loss(model1, model2), elbo.loss(model2, model1))
        self.assertAlmostEqual(elbo.loss(model2, model3), elbo.loss(model3, model2))
        self.assertNotEqual(elbo.loss(model3, model4), elbo.loss(model4, model3))

        self.assertTrue(is_density_equal(model1, model1, num_samples=num_samples))
        self.assertFalse(is_density_equal(model1, model2, num_samples=num_samples))
        self.assertTrue(is_density_equal(model2, model3, num_samples=num_samples))
        self.assertFalse(is_density_equal(model3, model4, num_samples=num_samples))

        self.assertTrue(is_intervention_density_equal(model1,
                                     model1, intervention={"x": 0}, num_samples=num_samples))
        self.assertTrue(is_intervention_density_equal(model1,model1, intervention={"x": 1}, num_samples=num_samples))
        self.assertFalse(is_intervention_density_equal(model1, model2, intervention={"x": 0}, num_samples=num_samples))
        self.assertTrue(is_intervention_density_equal(model2, model3, intervention={"x": 0}, num_samples=num_samples))
        self.assertTrue(is_intervention_density_equal(model3, model4, intervention={"x": 0}, num_samples=num_samples))
        self.assertFalse(is_intervention_density_equal(model3, model4, intervention={"x": 1}, num_samples=num_samples))
        #self.assertTrue(is_density_equal(model5, model6))

        #self.assertFalse(is_density_equal(do(model5, intervention={'y': 0 }),
        #                              do(model6, intervention={'y': 0 })))
        # The marginals are equal, but not the density.
        self.assertFalse(is_density_equal(model7, model8, num_samples=num_samples))
        self.assertFalse(is_intervention_density_equal(model7, model8, intervention={'x': 1}, num_samples=num_samples))
        self.assertTrue(is_density_equal(model9, model10, num_samples=num_samples))
        self.assertFalse(is_intervention_density_equal(model9, model10, intervention={'x': 1}, num_samples=num_samples))
