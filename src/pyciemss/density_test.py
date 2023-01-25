from pyro.infer import  Trace_ELBO
from pyro import do
import numpy as np
__all__ = ['is_density_equal',
           'is_intervention_density_equal']

def is_density_equal(model1: callable , model2: callable, num_samples:int=1000):
    """
    Test the density of two models.

    Args: model1: The first model.
          model2: The second model.
          num_samples: The number of samples to use.
    Returns: True if the density of the two models is the same.
    """
    elbo = Trace_ELBO(num_particles=num_samples, vectorize_particles=False)

    # compare the density of the two models
    return np.allclose( elbo.loss(model1, model2), elbo.loss(model2, model1), atol=1e-6)

def is_intervention_density_equal( model1: callable, model2: callable, intervention: dict, num_samples:int=1000):
    """Test the density of two models after intervention.

            Args: model1: The first model.
                    model2: The second model.
                    intervention: The intervention.
                    num_samples: The number of samples to use.
            Returns: True if the density of the two models is the same after intervention.
    """

    return is_density_equal(do(model1, intervention), do(model2, intervention), num_samples)
