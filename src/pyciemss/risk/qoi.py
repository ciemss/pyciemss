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


def fraction_infected(dataCube) -> np.ndarray:
    # return np.elementwise_division(dataCube["I_obs"], dataCube["N"])
    raise NotImplementedError


def threshold_exceedance(dataCube, threshold: float=0., contexts: list=None):
    '''
    # TODO: extend to handle multiple contexts
    '''
    if contexts is not None:
        dataCube = dataCube[contexts[0]].detach().numpy()
    
    # Return how many samples exceeded the threshold at ANY point
    return np.any(dataCube>=threshold, axis=1).astype(int)


# def total_infections_SIDARTHE(dataCube: np.ndarray, contexts: list=None) -> np.ndarray:
#     '''
#     Return samples of total infections from SIDARTHE model with dimension (nreplicates,).
#     dataCube is is the output from a Pyro Predictive object.
#     dataCube[VARIABLE] is expected to have dimension (nreplicates, ntimepoints)
#     '''
#     # Extract specific context response to compute on.
#     # TODO: check contexts inputs
#     # TODO: check shape of total_infections
#     total_infections = np.zeros(len(contexts), len(dataCube[contexts[0]]))
#     print(total_infections.shape)
#     if contexts is not None:
#         for i in range(len(contexts)):
#             total_infections[i,:] = dataCube[contexts[i]].detach().numpy()
#     print(total_infections.shape, np.sum(total_infections, axis=1).shape)
#     abc
#     return np.sum(total_infections, axis=1)


def max_total_infections_SIDARTHE(dataCube: np.ndarray, contexts: list=None) -> np.ndarray:
    '''
    Return samples of maximum total infections from SIDARTHE model in a given timeframe.
    dataCube is is the output from a Pyro Predictive object.
    dataCube[VARIABLE] is expected to have dimension (nreplicates, ntimepoints)
    '''
    # Extract specific context response to compute on.
    # TODO: check contexts inputs
    # TODO: check shape of total_infections
    # print(contexts, len(contexts))
    # total_infections = np.zeros((len(contexts), len(dataCube[contexts[0]])))
    # print(total_infections.shape)
    if contexts is not None:
        # for i in range(len(contexts)):
        total_infections = dataCube[contexts[0]].detach().numpy()
    # print(total_infections.shape, np.max(total_infections, axis=1).shape)
    return np.max(total_infections, axis=1)
