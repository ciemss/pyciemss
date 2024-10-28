import warnings
from typing import Dict, Optional, Union
from pyciemss.compiled_dynamics import get_name
import mira
import mira.metamodel
import networkx as nx
import pyro
import torch
import sympytorch
from mira.metamodel.utils import safe_parse_expr, SympyExprStr

ParameterDict = Dict[str, Union[torch.Tensor, SympyExprStr]]


def sort_mira_dependencies(src: mira.metamodel.TemplateModel) -> list:
    """
    Sort the model parameters of a MIRA TemplateModel by their distribution parameter dependencies.

    Parameters
    ----------
    src : mira.metamodel.TemplateModel
        The MIRA TemplateModel to sort.

    Returns
    -------
    list
        A list of parameter names in the order in which they must be evaluated.
    """
    dependencies = nx.DiGraph()
    for param_name, param_info in src.parameters.items():
        param_dist = getattr(param_info, "distribution", None)
        if param_dist is None:
            dependencies.add_node(param_name)
        else:
            for k, v in param_dist.parameters.items():
                # Check to see if the distribution parameters are sympy expressions
                # and add their free symbols to the dependency graph
                if isinstance(v, mira.metamodel.utils.SympyExprStr):
                    for free_symbol in v.free_symbols:
                        dependencies.add_edge(str(free_symbol), str(param_name))
    return  list(nx.topological_sort(dependencies))

def safe_sympytorch_parse_expr(expr: SympyExprStr, local_dict: Optional[Dict]) -> torch.Tensor:
    """
    Converts a sympy expression to a PyTorch tensor.

    Parameters
    ----------
    expr : SympyExprStr
        The sympy expression to convert to a PyTorch tensor.
    local_dict : Optional[Dict]
        A dictionary of free symbols and their variables to use in the sympy expression."""
    return sympytorch.SymPyModule(expressions=[expr.args[0]])(**local_dict).squeeze()

def mira_uniform_to_pyro(parameters: ParameterDict) -> pyro.distributions.Distribution:
    """
    Converts MIRA uniform distribution parameters to Pyro distribution.

    Parameters
    ----------
    parameters : ParameterDict
        Dictionary containing the parameters for the MIRA uniform distribution.

    Returns
    -------
    pyro.distributions.Distribution
        Pyro uniform distribution with specified lower and upper bounds.
    """
    low = parameters["minimum"]
    high = parameters["maximum"]
    return pyro.distributions.Uniform(low=low, high=high)


def mira_normal_to_pyro(parameters: ParameterDict) -> pyro.distributions.Distribution:
    """
    Converts MIRA normal distribution parameters to Pyro distribution.

    Parameters
    ----------
    parameters : ParameterDict
        Dictionary containing the parameters for the MIRA normal distribution.
        The parameters should contain one of the following sets of keys:
            - 'mean' and 'stdev'
            - 'mean' and 'variance'
            - 'mean' and 'precision'

    Returns
    -------
    pyro.distributions.Distribution
        Pyro normal distribution with specified mean and standard deviation.
    """
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
    """
    Converts MIRA lognormal distribution parameters to Pyro distribution.

    Parameters
    ----------
    parameters : ParameterDict
        Dictionary containing the parameters for the MIRA lognormal distribution.
        The parameters should contain one of the following sets of keys:
            - 'meanLog' and 'stdevLog'
            - 'meanLog' and 'varLog'

    Returns
    -------
    pyro.distributions.Distribution
        Pyro lognormal distribution with specified mean of the logarithm and standard deviation of the logarithm.
    """
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
    """
    Converts MIRA beta distribution parameters to Pyro distribution.

    Parameters
    ----------
    parameters : ParameterDict
        Dictionary containing the parameters for the MIRA beta distribution.
        The parameters should contain the following keys:
            - 'alpha' or 'concentration1'
            - 'beta' or 'concentration0'

    Returns
    -------
    pyro.distributions.Distribution
        Pyro Beta distribution with specified first and second shape parameters.
    """
    if "alpha" in parameters.keys():
        concentration1 = parameters["alpha"]
    elif "concentration1" in parameters.keys():
        concentration1 = parameters["concentration1"]
    else:
        raise ValueError(
            "MIRA Beta distribution requires 'alpha' or 'concentration1' parameter"
        )
    if "beta" in parameters.keys():
        concentration0 = parameters["beta"]
    elif "concentration0" in parameters.keys():
        concentration0 = parameters["concentration0"]
    else:
        raise ValueError(
            "MIRA Beta distribution requires 'beta' or 'concentration0' parameter"
        )
    return pyro.distributions.Beta(concentration1=concentration1, concentration0=concentration0)


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
    """
    Converts MIRA InverseGamma distribution parameters to Pyro distribution.

    Parameters
    ----------
    parameters : ParameterDict
        Dictionary containing the parameters for the MIRA InverseGamma distribution.
        The parameters should contain one of the following sets of keys:
            - 'shape' or 'alpha' or 'concentration'
            - 'rate' or 'scale' or 'beta'

    Returns
    -------
    pyro.distributions.Distribution
        Pyro InverseGamma distribution with specified shape and rate.
    """
    if "shape" in parameters.keys():
        concentration = parameters["shape"]
    elif "concentration" in parameters.keys():
        concentration = parameters["concentration"]
    elif "alpha" in parameters.keys():
        concentration = parameters["alpha"]
    else:
        raise ValueError(
            "MIRA InverseGamma distribution requires 'shape' or 'concentration' or 'alphaparameter"
        )
    if "rate" in parameters.keys():
        rate = parameters["rate"]
    elif "scale" in parameters.keys():
        rate = parameters["scale"]
    elif "beta" in parameters.keys():
        rate = parameters["beta"]
    else:
        raise ValueError(
            "MIRA InverseGamma distribution requires 'rate' or 'scale' or 'beta' parameter")      
    return pyro.distributions.InverseGamma(
        concentration=concentration, rate=rate
    )


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
    "InverseGamma1",
    "Normal1",
    "Normal2",
    "Normal3",
]

def mira_distribution_to_pyro(
    mira_dist: mira.metamodel.template_model.Distribution,
    free_symbols=Optional[Dict]
) -> pyro.distributions.Distribution:
    if mira_dist.type not in _MIRA_TO_PYRO.keys():
        raise NotImplementedError(
            f"Conversion from MIRA distribution type {mira_dist.type} to Pyro distribution is not implemented."
        )

    if mira_dist.type not in _TESTED_DISTRIBUTIONS:
        warnings.warn(
            f"Conversion from MIRA distribution type {mira_dist.type} to Pyro distribution has not been tested."
        )

    parameters = {
        k: safe_sympytorch_parse_expr(v, local_dict=free_symbols) 
        if isinstance(v, SympyExprStr) 
        else torch.as_tensor(v) 
        for k, v in mira_dist.parameters.items()
    }

    return _MIRA_TO_PYRO[mira_dist.type](parameters)
