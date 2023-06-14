import numpy as np

def scenario2dec_nday_average(dataCube, contexts: list=['I_obs'], ndays: int=7) -> np.ndarray:
    '''
    Return estimate of last n-day average of each sample.
    dataCube is is the output from a Pyro Predictive object.
    dataCube[VARIABLE] is expected to have dimension (nreplicates, ntimepoints)
    Note: last ndays timepoints is assumed to represent last n-days of simulation.
    '''
    if contexts is not None:
        if contexts[0] not in dataCube and contexts[0]=='I_obs':
            dataCube['I_obs'] = dataCube['I_sol'] + dataCube['I_v_sol']    # TODO: This is too specific and possibly needs to be changed
            dataQoI = dataCube[contexts[0]].detach().numpy()
        else:
            dataQoI = dataCube[contexts[0]].detach().numpy()

    return np.mean(dataQoI[:,-ndays:], axis=1)

# def scenario2dec_nday_rolling_average(dataCube, tf: float=90-1., ndays: int=7, contexts: list=['I_obs']) -> np.ndarray:
#     '''
#     Return estimate of n-day average of each sample.
#     dataCube is is the output from a Pyro Predictive object.
#     dataCube[VARIABLE] is expected to have dimension (nreplicates, ntimepoints)
#     note: tf is set to be consistent with the tspan.
#     '''
#     # Extract specific context response to compute on.
#     if contexts is not None:
#         if contexts[0]=='I_obs':
#             dataCube['I_obs'] = dataCube['I_sol'] + dataCube['I_v_sol']    # TODO: This is too specific and needs to be changed
#             dataQoI = dataCube[contexts[0]].detach().numpy()
#         else:
#             dataQoI = dataCube[contexts[0]].detach().numpy()

#     ndayavg = dataQoI[:, int(tf)-ndays:int(tf)]
#     return np.mean(ndayavg, axis=1)


# def fraction_infected(dataCube) -> np.ndarray:
#     raise NotImplementedError


# def threshold_exceedance(dataCube, threshold: float=0., contexts: list=None):
#     '''
#     # TODO: extend to handle multiple contexts
#     '''
#     if contexts is not None:
#         dataCube = dataCube[contexts[0]].detach().numpy()
    
#     # Return how many samples exceeded the threshold at ANY point
#     return np.any(dataCube>=threshold, axis=1).astype(int)


# def max_total_infections_SIDARTHE(dataCube: np.ndarray, contexts: list=None) -> np.ndarray:
#     '''
#     Return samples of maximum total infections from SIDARTHE model in a given timeframe.
#     dataCube is is the output from a Pyro Predictive object.
#     dataCube[VARIABLE] is expected to have dimension (nreplicates, ntimepoints)
#     '''
#     # Extract specific context response to compute on.
#     # TODO: check contexts inputs
#     # TODO: check shape of total_infections
#     # print(contexts, len(contexts))
#     # total_infections = np.zeros((len(contexts), len(dataCube[contexts[0]])))
#     # print(total_infections.shape)
#     if contexts is not None:
#         # for i in range(len(contexts)):
#         total_infections = dataCube[contexts[0]].detach().numpy()
#     # print(total_infections.shape, np.max(total_infections, axis=1).shape)
#     return np.max(total_infections, axis=1)