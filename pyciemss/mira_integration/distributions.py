import warnings
from typing import Dict

import mira.metamodel
import pyro
import torch

ParameterDict = Dict[str, torch.Tensor]


def mira_uniform_to_pyro(parameters: ParameterDict) -> pyro.distributions.Distribution:
    low = parameters["minimum"]
    high = parameters["maximum"]
    return pyro.distributions.Uniform(low=low, high=high)


def mira_normal_to_pyro(parameters: ParameterDict) -> pyro.distributions.Distribution:
    if "mean" in parameters.keys():
        loc = parameters["mean"]
    if "stdev" in parameters.keys():
        scale = parameters["stdev"]
    elif "variance" in parameters.keys():
        scale = parameters["variance"] ** 0.5
    elif "precision" in parameters.keys():
        scale = parameters["precision"] ** -0.5

    return pyro.distributions.Normal(loc=loc, scale=scale)


def mira_lognormal_to_pyro(
    parameters: ParameterDict,
) -> pyro.distributions.Distribution:
    if "meanLog" in parameters.keys():
        loc = parameters["meanLog"]
    if "stdevLog" in parameters.keys():
        scale = parameters["stdevLog"]
    elif "varLog" in parameters.keys():
        scale = parameters["varLog"] ** 0.5

    return pyro.distributions.LogNormal(loc=loc, scale=scale)


# Provide either probs or logits, not both
def mira_bernoulli_to_pyro(
    parameters: ParameterDict,
) -> pyro.distributions.Distribution:
    if "probability" in parameters.keys():
        probs = parameters["probability"]
        logits = None
    elif "logitProbability" in parameters.keys():
        probs = None
        logits = parameters["logitProbability"]

    return pyro.distributions.Bernoulli(probs=probs, logits=logits)


def mira_beta_to_pyro(parameters: ParameterDict) -> pyro.distributions.Distribution:
    return pyro.distributions.Beta(alpha=parameters["alpha"], beta=parameters["beta"])


def mira_betabinomial_to_pyro(
    parameters: ParameterDict,
) -> pyro.distributions.Distribution:
    concentration1 = parameters["alpha"]
    concentration0 = parameters["beta"]
    total_count = parameters["numberOfTrials"]

    return pyro.distributions.BetaBinomial(
        concentration1=concentration1,
        concentration0=concentration0,
        total_count=total_count,
    )


def mira_binomial_to_pyro(parameters: ParameterDict) -> pyro.distributions.Distribution:
    total_count = parameters["numberOfTrials"]
    if "probability" in parameters.keys():
        probs = parameters["probability"]
        logits = None
    elif "logitProbability" in parameters.keys():
        probs = None
        logits = parameters["logitProbability"]

    return pyro.distributions.Binomial(
        total_count=total_count, probs=probs, logits=logits
    )


def mira_cauchy_to_pyro(parameters: ParameterDict) -> pyro.distributions.Distribution:
    loc = parameters["location"]
    scale = parameters["scale"]
    return pyro.distributions.Cauchy(loc=loc, scale=scale)


def mira_chisquared_to_pyro(
    parameters: ParameterDict,
) -> pyro.distributions.Distribution:
    df = parameters["degreesOfFreedom"]
    return pyro.distributions.Chi2(df=df)


def mira_dirichlet_to_pyro(
    parameters: ParameterDict,
) -> pyro.distributions.Distribution:
    concentration = parameters["concentration"]
    return pyro.distributions.Dirichlet(concentration=concentration)


def mira_exponential_to_pyro(
    parameters: ParameterDict,
) -> pyro.distributions.Distribution:
    if "rate" in parameters.keys():
        rate = parameters["rate"]
    elif "mean" in parameters.keys():
        rate = 1.0 / parameters["mean"]
    return pyro.distributions.Exponential(rate=rate)


def mira_gamma_to_pyro(parameters: ParameterDict) -> pyro.distributions.Distribution:
    if "shape" in parameters.keys():
        concentration = parameters["shape"]
    if "scale" in parameters.keys():
        rate = 1.0 / parameters["scale"]
    elif "rate" in parameters.keys():
        rate = parameters["rate"]
    return pyro.distributions.Gamma(concentration=concentration, rate=rate)


def mira_inversegamma_to_pyro(
    parameters: ParameterDict,
) -> pyro.distributions.Distribution:
    raise NotImplementedError(
        "Conversion from MIRA InverseGamma distribution to Pyro distribution is not implemented."
    )
    # TODO: Map parameters to Pyro distribution parameters


def mira_gumbel_to_pyro(parameters: ParameterDict) -> pyro.distributions.Distribution:
    loc = parameters["location"]
    scale = parameters["scale"]
    return pyro.distributions.Gumbel(loc=loc, scale=scale)


def mira_laplace_to_pyro(parameters: ParameterDict) -> pyro.distributions.Distribution:
    if "location" in parameters.keys():
        loc = parameters["location"]
    elif "mu" in parameters.keys():
        loc = parameters["mu"]

    if "scale" in parameters.keys():
        scale = parameters["scale"]
    elif "tau" in parameters.keys():
        scale = 1.0 / parameters["tau"]

    return pyro.distributions.Laplace(loc=loc, scale=scale)


def mira_paretotypeI_to_pyro(
    parameters: ParameterDict,
) -> pyro.distributions.Distribution:
    scale = parameters["scale"]
    alpha = parameters["shape"]
    return pyro.distributions.Pareto(scale=scale, alpha=alpha)


def mira_poisson_to_pyro(parameters: ParameterDict) -> pyro.distributions.Distribution:
    rate = parameters["rate"]
    return pyro.distributions.Poisson(rate=rate)


def mira_studentt_to_pyro(parameters: ParameterDict) -> pyro.distributions.Distribution:
    if "mean" in parameters.keys():
        loc = parameters["mean"]
    elif "location" in parameters.keys():
        loc = parameters["location"]
    else:
        loc = torch.tensor(0.0)

    if "scale" in parameters.keys():
        scale = parameters["scale"]
    else:
        scale = torch.tensor(1.0)

    df = parameters["degreesOfFreedom"]

    return pyro.distributions.StudentT(df=df, loc=loc, scale=scale)


def mira_weibull_to_pyro(parameters: ParameterDict) -> pyro.distributions.Distribution:
    if "scale" in parameters.keys():
        scale = parameters["scale"]
    elif "lambda" in parameters.keys():
        scale = parameters["lambda"]

    concentration = parameters["shape"]

    return pyro.distributions.Weibull(scale=scale, concentration=concentration)


# Key - MIRA distribution type : str
# Value - MIRA -> Pyro function : Callable[[ParameterDict], pyro.distributions.Distribution]
# See https://github.com/indralab/mira/blob/main/mira/dkg/resources/probonto.json for MIRA distribution types
_MIRA_TO_PYRO = {
    "Uniform1": mira_uniform_to_pyro,
    "StandardUniform1": mira_uniform_to_pyro,
    "StandardNormal1": mira_normal_to_pyro,
    "Normal1": mira_normal_to_pyro,
    "Normal2": mira_normal_to_pyro,
    "Normal3": mira_normal_to_pyro,
    "LogNormal1": mira_lognormal_to_pyro,
    "LogNormal2": mira_lognormal_to_pyro,
    "Bernoulli1": mira_bernoulli_to_pyro,
    "Bernoulli2": mira_bernoulli_to_pyro,
    "Beta1": mira_beta_to_pyro,
    "BetaBinomial1": mira_betabinomial_to_pyro,
    "Binomial1": mira_binomial_to_pyro,
    "Binomial2": mira_binomial_to_pyro,
    "Cauchy1": mira_cauchy_to_pyro,
    "ChiSquared1": mira_chisquared_to_pyro,
    "Dirichlet1": mira_dirichlet_to_pyro,
    "Exponential1": mira_exponential_to_pyro,
    "Exponential2": mira_exponential_to_pyro,
    "Gamma1": mira_gamma_to_pyro,
    "Gamma2": mira_gamma_to_pyro,
    "InverseGamma1": mira_inversegamma_to_pyro,
    "Gumbel1": mira_gumbel_to_pyro,
    "Laplace1": mira_laplace_to_pyro,
    "Laplace2": mira_laplace_to_pyro,
    "ParetoTypeI1": mira_paretotypeI_to_pyro,
    "Poisson1": mira_poisson_to_pyro,
    "StudentT1": mira_studentt_to_pyro,
    "StudentT2": mira_studentt_to_pyro,
    "StudentT3": mira_studentt_to_pyro,
    "Weibull1": mira_weibull_to_pyro,
    "Weibull2": mira_weibull_to_pyro,
}

_TESTED_DISTRIBUTIONS = [
    "Uniform1",
    "StandardUniform1",
    "StandardNormal1",
    "Normal1",
    "Normal2",
    "Normal3",
]


def mira_distribution_to_pyro(
    mira_dist: mira.metamodel.template_model.Distribution,
) -> pyro.distributions.Distribution:
    if mira_dist.type not in _MIRA_TO_PYRO.keys():
        raise NotImplementedError(
            f"Conversion from MIRA distribution type {mira_dist.type} to Pyro distribution is not implemented."
        )

    if mira_dist.type not in _TESTED_DISTRIBUTIONS:
        warnings.warn(
            f"Conversion from MIRA distribution type {mira_dist.type} to Pyro distribution has not been tested."
        )

    parameters = {k: torch.as_tensor(v) for k, v in mira_dist.parameters.items()}

    return _MIRA_TO_PYRO[mira_dist.type](parameters)
