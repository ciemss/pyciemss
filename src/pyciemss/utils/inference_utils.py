import torch
import pyro
from causal_pyro.query.do_messenger import do
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.poutine import trace, replay
import numpy as np

from math import isclose

__all__ = ['is_density_equal',
           'is_intervention_density_equal',
           'get_tspan',
           'state_flux_constraint',
           'run_inference']

def run_inference(model,
                guide,
                initial_state,
                tspan,
                data,
                optim=Adam({'lr': 0.03}),
                loss_f=Trace_ELBO(num_particles=1),
                num_iterations=250,
                verbose=False
                ):
    '''
    Run stochastic variational inference.
    This is just a very thin abstraction around Pyro's SVI class.
    '''

    svi = SVI(model, guide, optim, loss=loss_f)

    pyro.clear_param_store()

    for j in range(num_iterations):
        # calculate the loss and take a gradient step
        loss = svi.step(initial_state, tspan, data)
        if verbose:
            if j % 25 == 0:
                print("[iteration %04d] loss: %.4f" % (j + 1, loss))

def state_flux_constraint(S, flux):
    '''
    Check the constraint that the state value is always positive.
    Check the constraint that the state flux is always negative.
    If either of these conditions do not hold, set the resulting state flux to be 0.
    '''
    satisfied_index = torch.logical_and(S > 0, flux > 0)
    return torch.where(satisfied_index, flux, torch.zeros_like(flux))

def get_tspan(start, end, steps):
    '''
    Thin wrapper around torch.linspace.
    '''
    return torch.linspace(float(start), float(end), steps)

def is_density_equal(model1: callable , model2: callable, *args, **kwargs):
    """
    Test the equality of the density of two models for samples drawn from their priors.

    Args: model1: The first model.
          model2: The second model.
          num_samples: The number of samples to use.
    Returns: True if the density of the two models is the same.
    """
    
    tr1 = trace(model1).get_trace(*args, **kwargs)
    tr2 = trace(model2).get_trace(*args, **kwargs)

    d_m1_tr1 = trace(replay(model1, trace=tr1)).get_trace(*args, **kwargs).log_prob_sum()
    d_m2_tr1 = trace(replay(model2, trace=tr1)).get_trace(*args, **kwargs).log_prob_sum()
    d_m1_tr2 = trace(replay(model1, trace=tr2)).get_trace(*args, **kwargs).log_prob_sum()
    d_m2_tr2 = trace(replay(model2, trace=tr2)).get_trace(*args, **kwargs).log_prob_sum()

    rel_tol = 1e-2

    return isclose(d_m1_tr1, d_m2_tr1, rel_tol=rel_tol) and isclose(d_m1_tr2, d_m2_tr2, rel_tol=rel_tol)


    elbo = Trace_ELBO(num_particles=num_samples, vectorize_particles=False)

    print(elbo.loss(model1, model2, *args, **kwargs), elbo.loss(model2, model1, *args, **kwargs))

    # compare the density of the two models
    return np.allclose(elbo.loss(model1, model2, *args, **kwargs), elbo.loss(model2, model1, *args, **kwargs), atol=1e-6)

def is_intervention_density_equal( model1: callable, model2: callable, intervention: dict, *args, **kwargs):
    """Test the density of two models after intervention.

            Args: model1: The first model.
                    model2: The second model.
                    intervention: The intervention.
                    num_samples: The number of samples to use.
            Returns: True if the density of the two models is the same after intervention.
    """

    return is_density_equal(do(model1, intervention), do(model2, intervention), *args, **kwargs)
