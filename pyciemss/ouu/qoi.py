import numpy as np
from typing import List


def obs_nday_average_qoi(
    dataCube, contexts: List, ndays: int = 7
) -> np.ndarray:
    """
    Return estimate of last n-day average of each sample.
    dataCube is is the output from a Pyro Predictive object.
    dataCube[VARIABLE] is expected to have dimension (nreplicates, ntimepoints)
    Note: last ndays timepoints is assumed to represent last n-days of simulation.
    """
    if contexts is not None:
            dataQoI = dataCube[contexts[0]].detach().numpy()

    return np.mean(dataQoI[:, -ndays:], axis=1)
