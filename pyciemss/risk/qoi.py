import numpy as np

def rolling_average_infected(I, Iv=None, tf=90, ndays=7, dt=1.):
    '''
    Return estimate of n-day average of total infections.
    '''
    # Estimate n-day average of cases
    if Iv is None:
        I_ndays = I[:, int(tf/dt)-ndays+1:int(tf/dt)+1]
        return np.mean(I_ndays, axis=1)
    else:
        I_ndays = I[:, int(tf/dt)-ndays+1:int(tf/dt)+1]
        Iv_ndays = Iv[:, int(tf/dt)-ndays+1:int(tf/dt)+1]
        return np.mean(I_ndays + Iv_ndays, axis=1)
