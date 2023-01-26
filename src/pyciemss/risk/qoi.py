import numpy as np
from pyciemss.risk.risk_measures import pof

def nday_rolling_average(samples: np.ndarray, tf=90., ndays=7, dt=1.) -> np.ndarray:
    '''
    Return estimate of n-day average of samples.
    samples is a numpy array of shpae (num_samples, num_times_steps).
    '''
    samples_ndays = samples[int(tf/dt)-ndays+1:int(tf/dt)+1, :]
    return np.mean(samples_ndays, axis=0)


# TODO: rewrite this so it's not pseudocode
def fraction_infected(samples: np.ndarray) -> np.ndarray:
    return np.elementwise_division(samples["I_obs"], samples["N"])


def probability_of_exceedence(samples, threshold, contexts: list=None):
    '''
    Thin wrapper around risk.risk_measures.pof
    # TODO: extend to handle multiple contexts
    '''
    if contexts is not None:
        samples = samples[contexts[0]].detach().numpy()
    return pof(samples, threshold)
