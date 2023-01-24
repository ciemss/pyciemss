from pyciemss import test_density
import pyro
import unittest
from pyro.infer import  Trace_ELBO

class TestDensityTest(unittest.TestCase):
    """Tests for comparing the density of two models."""

    def test_density_test(self):
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


        # Define the number of samples
        num_samples = 1000
        # Define the loss function
        elbo = Trace_ELBO(num_particles=num_samples, vectorize_particles=False)

        # Sample from both models
        self.assertAlmostEqual( elbo.loss(model1, model1),elbo.loss(model1, model1))
        self.assertNotEqual(elbo.loss(model1, model2), elbo.loss(model2, model1))
        self.assertAlmostEqual(elbo.loss(model2, model3), elbo.loss(model3, model2))
        self.assertNotEqual(elbo.loss(model3, model4), elbo.loss(model4, model3))

        self.assertTrue(test_density(model1, model1))
        self.assertFalse(test_density(model1, model2))
        self.assertTrue(test_density(model2, model3))
        self.assertFalse(test_density(model3, model4))

        self.assertTrue(test_density(pyro.do(model1, data={"x": 0}), 
                                     pyro.do(model1, data={"x": 0})))
        self.assertFalse(test_density(pyro.do(model1, data={"x": 0}),
                                      pyro.do(model1, data={"x": 1})))
        self.assertFalse(test_density(pyro.do(model1, data={"x": 0}),
                                      pyro.do(model2, data={"x": 0})))
        self.assertTrue(test_density(pyro.do(model2, data={"x": 0}),
                                     pyro.do(model3, data={"x": 0})))   
        self.assertTrue(test_density(pyro.do(model3, data={"x": 0}),
                                     pyro.do(model4, data={"x": 0}))) 
        self.assertFalse(test_density(pyro.do(model3, data={"x": 1}),
                                      pyro.do(model4, data={"x": 1})))          