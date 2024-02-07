import numpy as np


def scenario2dec_nday_average(
    dataCube, contexts: list = ["I_obs"], ndays: int = 7
) -> np.ndarray:
    """
    Return estimate of last n-day average of each sample.
    dataCube is is the output from a Pyro Predictive object.
    dataCube[VARIABLE] is expected to have dimension (nreplicates, ntimepoints)
    Note: last ndays timepoints is assumed to represent last n-days of simulation.
    """
    if contexts is not None:
        if contexts[0] not in dataCube and contexts[0] == "I_obs":
            dataCube["I_obs"] = (
                dataCube["I_sol"] + dataCube["I_v_sol"]
            )  # TODO: This is too specific and possibly needs to be changed
            dataQoI = dataCube[contexts[0]].detach().numpy()
        else:
            dataQoI = dataCube[contexts[0]].detach().numpy()

    return np.mean(dataQoI[:, -ndays:], axis=1)
