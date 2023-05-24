from pyro.distributions import Beta, TransformedDistribution
from torch.distributions import constraints
from torch.distributions.transforms import AffineTransform


class ScaledBeta(TransformedDistribution):
    r"""
    Creates a scaled beta distribution parameterized by
    :attr:`mean`, :attr:`max`, and :attr:`pseudocount` where::
        scaled_mean = mean / max
        X ~ Beta(scaled_mean * pseudocount, (1 - scaled_mean) * pseudocount)
        Y = X * max ~ ScaledBeta(mean, max, pseudocount)
    Args:
        mean (float or Tensor): mean of the distribution
        max (float or Tensor): maximum value of the distribution
        pseudocount (float or Tensor): pseudocount for the distribution ( == a + b in the underlying Beta distribution)
    """
    # TODO: Fix constraints
    # We really want 0 <= mean <= max and support = [0, max]. Can we express that?
    has_rsample = True

    def __init__(self, _mean, _max, pseudocount, validate_args=None):
        self._mean = _mean
        self._max = _max
        self._pseudocount = pseudocount
        scaled_mean = self._mean / self._max
        self._scaled_mean = scaled_mean
        base_dist = Beta(
            scaled_mean * pseudocount,
            (1 - scaled_mean) * pseudocount,
            validate_args=validate_args,
        )
        super().__init__(
            base_dist, AffineTransform(0, _max), validate_args=validate_args
        )

    @property
    def mean(self):
        return self.base_dist.mean() * self.max

    @property
    def max(self):
        return self._max

    @property
    def variance(self):
        return self.base_dist.variance() * self._max**2
