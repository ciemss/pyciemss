import pyro

from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

def run_inference(model, 
                guide, 
                initial_state,
                tspan, 
                data, 
                optim=Adam({'lr': 0.03}), 
                loss=Trace_ELBO(num_particles=1), 
                num_iterations=250, 
                verbose=False
                ):

    svi = SVI(model, guide, optim, loss=loss)

    pyro.clear_param_store()

    for j in range(num_iterations):
        # calculate the loss and take a gradient step
        loss = svi.step(initial_state, tspan, data)
        if verbose:
            if j % 5 == 0:
                print("[iteration %04d] loss: %.4f" % (j + 1, loss))