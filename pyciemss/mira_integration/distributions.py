from typing import Dict

import mira.metamodel
import pyro
from pyro.distributions.torch_distribution import TorchDistributionMixin


def mira_uniform_to_pyro(parameters: Dict[str, float]) -> TorchDistributionMixin:
    minimum = parameters["minimum"]
    maximum = parameters["maximum"]
    return pyro.distributions.Uniform(minimum, maximum)  # type: ignore[attr-defined]


def mira_normal_to_pyro(parameters: Dict[str, float]) -> TorchDistributionMixin:
    if "mean" in parameters.keys():
        mean = parameters["mean"]

    if "stdev" in parameters.keys():
        std = parameters["stdev"]
    elif "variance" in parameters.keys():
        std = parameters["variance"] ** 0.5
    elif "precision" in parameters.keys():
        std = parameters["precision"] ** -0.5

    #  Pyro distributions are thing wrappers around torch distributions.
    #  See https://pytorch.org/docs/stable/generated/torch.normal.html
    return pyro.distributions.Normal(mean, std)  # type: ignore[attr-defined]


# TODO: Add lognormal, beta, gamma, etc.

# Key - MIRA distribution type : str
# Value - MIRA -> Pyro function : Callable[[Dict[str, float]], pyro.distributions.Distribution]
# See https://github.com/indralab/mira/blob/main/mira/dkg/resources/probonto.json for MIRA distribution types
_MIRA_TO_PYRO = {
    "Uniform1": mira_uniform_to_pyro,
    "StandardUniform1": mira_uniform_to_pyro,
    "StandardNormal1": mira_normal_to_pyro,
    "Normal1": mira_normal_to_pyro,
    "Normal2": mira_normal_to_pyro,
    "Normal3": mira_normal_to_pyro,
}


def mira_distribution_to_pyro(
    mira_dist: mira.metamodel.template_model.Distribution,
) -> TorchDistributionMixin:
    if mira_dist.type not in _MIRA_TO_PYRO.keys():
        raise NotImplementedError(
            f"Conversion from MIRA distribution type {mira_dist.type} to Pyro distribution not implemented."
        )

    return _MIRA_TO_PYRO[mira_dist.type](mira_dist.parameters)
