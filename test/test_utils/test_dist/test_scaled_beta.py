import torch

from pyciemss.utils.distributions import ScaledBeta


def test_scaled_beta():
    dist = ScaledBeta(4, 5, 10)
    basedist = dist.base_dist

    samples = dist.sample((10000,))

    assert torch.isclose(samples.mean(), torch.tensor(4.0), atol=0.1)
    assert 4.5 < torch.max(samples) <= 5
    assert torch.all(
        torch.isclose(
            basedist.log_prob(samples / 5) - dist.log_prob(samples),
            torch.log(torch.tensor(5)),
        )
    )
