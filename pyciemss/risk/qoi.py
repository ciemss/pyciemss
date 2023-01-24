import numpy as np

def nday_rolling_average(samples: np.ndarray, tf=90., ndays=7, dt=1.) -> np.ndarray:
    '''
    Return estimate of n-day average of samples.
    samples is a numpy array of shpae (num_samples, num_times_steps).
    '''
    samples_ndays = samples[int(tf/dt)-ndays+1:int(tf/dt)+1, :]
    return np.mean(samples_ndays, axis=0)
