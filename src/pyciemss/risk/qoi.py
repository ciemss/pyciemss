import numpy as np

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


# TODO: rewrite this so it's not pseudocode
def exceedence_threshold(samples, threshold):
    if np.any(samples < threshold):
        return 1
    else:
        return 0
