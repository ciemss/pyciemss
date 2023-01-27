import numpy as np
from pyciemss.risk.risk_measures import pof

def nday_rolling_average(dataCube: np.ndarray, tf: float=90-1., ndays: int=7, dt: float =1., contexts: list=None) -> np.ndarray:
    '''
    Return estimate of n-day average of samples.
    dataCube is is the output from a Pyro Predictive object.
    dataCube[VARIABLE] is expected to have dimension (nreplicates, ntimepoints)
    note: tf is set to be consistent with the tspan.
    '''
    # Extract specific context response to compute on.
    if contexts is not None:
        dataCube = dataCube[contexts[0]].detach().numpy()

    ndayavg = dataCube[:, int(tf/dt)-ndays:int(tf/dt)]
    return np.mean(ndayavg, axis=1)


# TODO: rewrite this so it's not pseudocode
def fraction_infected(dataCube) -> np.ndarray:
    return np.elementwise_division(dataCube["I_obs"], dataCube["N"])


def threshold_exceedence(dataCube, threshold: float, contexts: list=None):
    '''
    # TODO: extend to handle multiple contexts
    '''
    if contexts is not None:
        dataCube = dataCube[contexts[0]].detach().numpy()
    
    # Return how many samples exceeded the threshold at ANY point
    return np.any(dataCube >= threshold, axis=1).astype(int)
