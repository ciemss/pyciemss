""" Provides any probability distributions used by the model that aren't in numpy/cupy."""

import numpy as np
import scipy.special as sc


# TODO only works on cpu atm
# we'd need to implement betaincinv ourselves in cupy
def mPERT_sample(mu, a=0.0, b=1.0, gamma=4.0, var=None):
    """Provides a vectorized Modified PERT distribution.

    Parameters
    ----------
    mu : float, array_like
        Mean value for the PERT distribution.
    a : float, array_like
        Lower bound for the distribution.
    b : float, array_like
        Upper bound for the distribution.
    gamma : float, array_like
        Shape paramter.
    var : float, array_like, None
        Variance of the distribution. If var != None,
        gamma will be calcuated to meet the desired variance.

    Returns
    -------
    out : float, array_like
        Samples drawn from the specified mPERT distribution.
        Shape is the broadcasted shape of the the input parameters.

    """
    mu, a, b = np.atleast_1d(mu, a, b)
    if var is not None:
        gamma = (mu - a) * (b - mu) / var - 3.0
    alp1 = 1.0 + gamma * ((mu - a) / (b - a))
    alp2 = 1.0 + gamma * ((b - mu) / (b - a))
    u = np.random.random_sample(mu.shape)
    alp3 = sc.betaincinv(alp1, alp2, u)
    return (b - a) * alp3 + a


def truncnorm(xp, loc=0.0, scale=1.0, size=1, a_min=None, a_max=None):
    """Provides a vectorized truncnorm implementation that is compatible with cupy.

    The output is calculated by using the numpy/cupy random.normal() and
    truncted via rejection sampling. The interface is intended to mirror
    the scipy implementation of truncnorm.

    Parameters
    ----------
    xp : module


    Returns
    -------

    """
    ret = xp.random.normal(loc, scale, size)
    if a_min is None:
        a_min = -xp.inf
    if a_max is None:
        a_max = xp.inf

    while True:
        valid = (ret > a_min) & (ret < a_max)
        if valid.all():
            return ret
        ret[~valid] = xp.random.normal(loc, scale, ret[~valid].shape)
