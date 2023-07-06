from pyciemss.utils.distributions import mira_distribution_to_pyro
from mira.metamodel.template_model import Distribution
import pytest
import torch
import pyro

@pytest.mark.parametrize("mira_dist", [Distribution(type="Normal1", parameters={"mean": 0, "stdev": 2.}),
                                       Distribution(type="Normal2", parameters={"mean": 0, "variance": 4.}),
                                       Distribution(type="Normal3", parameters={"mean": 0, "precision": 0.25})])
def test_normal(mira_dist):

    true_dist = pyro.distributions.Normal(0, 2.)
    pyro_dist = mira_distribution_to_pyro(mira_dist)

    samples = pyro_dist.sample((10000,))

    assert type(pyro_dist) == type(true_dist) == pyro.distributions.Normal
    assert torch.isclose(samples.mean(), torch.tensor(0.0), atol=0.1)
    assert torch.all(torch.isclose(pyro_dist.log_prob(samples), true_dist.log_prob(samples)))

@pytest.mark.parametrize("mira_dist", [Distribution(type="Uniform1", parameters={"minimum": 0, "maximum": 2.}),
                                       Distribution(type="StandardUniform1", parameters={"minimum": 0, "maximum": 2.})])
def test_uniform(mira_dist):

    true_dist = pyro.distributions.Uniform(0, 2.)
    pyro_dist = mira_distribution_to_pyro(mira_dist)

    samples = pyro_dist.sample((10000,))

    assert type(pyro_dist) == type(true_dist) == pyro.distributions.Uniform
    assert torch.isclose(samples.mean(), torch.tensor(1.0), atol=0.1)
    assert 0 < torch.min(samples)
    assert torch.max(samples) <= 2
    assert torch.all(torch.isclose(pyro_dist.log_prob(samples), true_dist.log_prob(samples)))