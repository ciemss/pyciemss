import torch
import pyro

from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

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
    Enforce the constraint that the state value is always positive.
    Enforce the constraint the the state flux is always negative.
    If either of these conditions do not hold, set the resulting state flux to be 0.
    '''
    if S.item() < 0 or flux.item() < 0:
        return torch.zeros_like(flux)
    else:
        return flux


def elvis(first, last):
    '''
    Check if `first` value isnan(). If so, return `last`. Otherwise, return `first`.
    '''
    if first.isnan():
        return last
    else:
        return first

def get_tspan(start, end, steps):
    return torch.linspace(float(start), float(end), steps)