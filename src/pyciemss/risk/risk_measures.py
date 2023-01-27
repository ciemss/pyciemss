import numpy as np

def mean(samples):
    '''
    Thin wrapper around numpy.mean.
    '''
    return np.mean(samples)


def sample_variance(samples):
    '''
    Thin wrapper around numpy.var.
    '''
    return np.var(samples)


def robust(samples, eta=2.):
    '''
    Robust combination of mean and variance.
    '''
    return np.mean(samples) + eta*np.std(samples)


def pof(samples, threshold=0.):
    '''
    Probability of exceeding a threshold.
    '''
    return np.sum(samples >= threshold)/float(samples.shape[0])


def buffered_pof(samples, threshold=0.):
    '''
    Buffered probability of exceeding a threshold.
    '''
    nsamples = int(samples.shape[0])
    sorted_samples = np.sort(samples)[::-1]    # sort samples in descending order
    squantile = sorted_samples[0]
    prob_level = 1
    for i in range(nsamples-1):
        prob_level = prob_level - 1./nsamples
        squantile = (squantile*i + sorted_samples[i+1])/float(i+1)
        if squantile < threshold:
            break

    return 1. - prob_level


def alpha_quantile(samples, alpha=0.95):
    '''
    Upper alpha-quantile for a given alpha in [0,1]
    a.k.a value-at-risk (VaR).
    Thin wrapper for numpy.quantile.
    '''
    return np.quantile(samples, alpha)


def alpha_superquantile(samples, alpha=0.95):
    '''
    upper alpha-superquantile for a given alpha in [0,1] a.k.a conditional value-at-risk (CVaR), expected shortfall, average value-at-risk.
    '''
    nsamples = int(samples.shape[0])
    sorted_samples = np.sort(samples)[::-1]    # sort samples in descending order
    ka = int(np.ceil(nsamples*(1-alpha)))  # index for alpha-quantile
    qtile = sorted_samples[ka-1]
    return qtile + 1./(nsamples*(1-alpha))*np.sum(sorted_samples[0:ka-1]-qtile)
    