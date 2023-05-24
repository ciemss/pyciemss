import mira
import pyro
import pyro.distributions as dist
import torch
from torch import Tensor

from pyciemss.PetriNetODE.base import (
    MiraPetriNetODESystem,
    PetriNetODESystem,
    ScaledBetaNoisePetriNetODESystem,
    Solution,
    State,
    StateDeriv,
    Time,
    get_name,
)
from pyciemss.utils.distributions import ScaledBeta


class SIR_with_uncertainty(PetriNetODESystem):
    """SIR model built by hand to compare against the MIRA SIR model
    See https://github.com/ciemss/pyciemss/issues/144
    for more detail.
    """

    def __init__(
        self,
        N: int,
        beta: float,
        gamma: float,
        pseudocount: float = 1.0,
    ) -> None:
        """initialize total population beta and gamma parameters
        :param N: total population
        :param beta: infection rate
        :param gamma: recovery rate
        """
        super().__init__()
        self.total_population = N
        self.beta_prior = pyro.distributions.Uniform(max(0.9 * beta, 0.0), 1.1 * beta)
        self.gamma_prior = pyro.distributions.Uniform(
            max(0.9 * gamma, 0.0), 1.1 * gamma
        )
        self.pseudocount = pseudocount

    def create_var_order(self) -> dict[str, int]:
        """create the variable order for the state vector"""
        return {
            "susceptible_population": 0,
            "infected_population": 1,
            "immune_population": 2,
        }

    @pyro.nn.pyro_method
    def deriv(self, t: Time, state: State) -> StateDeriv:
        """compute the state derivative at time t
        :param t: time
        :param state: state vector
        :return: state derivative vector
        """
        assert torch.isclose(
            torch.as_tensor(sum(state)), self.total_population
        ), f"The sum of state variables {state} is not scaled to the total population {self.total_population}."

        S, I, R = state

        dSdt = -self.beta * S * I / self.total_population
        dIdt = self.beta * S * I / self.total_population - self.gamma * I
        dRdt = self.gamma * I
        return dSdt, dIdt, dRdt

    @pyro.nn.pyro_method
    def param_prior(self) -> None:
        """define the prior distributions for the parameters"""
        setattr(self, "beta", pyro.sample("beta", self.beta_prior))
        setattr(self, "gamma", pyro.sample("gamma", self.gamma_prior))

    @pyro.nn.pyro_method
    def observation_model(self, solution: Solution, var_name: str) -> None:
        """define the observation model for the given variable
        :param solution: solution of the ODE system
        :param var_name: variable name
        """
        mean = solution[var_name]
        pseudocount = self.pseudocount
        pyro.sample(
            var_name, ScaledBeta(mean, self.total_population, pseudocount).to_event(1)
        )

    def static_parameter_intervention(self, parameter: str, value: Tensor) -> None:
        """set a static parameter intervention
        :param parameter: parameter name
        :param value: parameter value
        """
        setattr(self, parameter, value)


class MIRA_SVIIvR(ScaledBetaNoisePetriNetODESystem):
    @pyro.nn.pyro_method
    def observation_model(self, solution: Solution, var_name: str) -> None:
        """In the observation model, I_obs is the sum of I and Iv."""
        if var_name == "I_obs":
            mean = solution["I"] + solution["I_v"]
        else:
            mean = solution[var_name]
        pyro.sample(
            var_name,
            ScaledBeta(mean, self.total_population, self.pseudocount * mean).to_event(
                1
            ),
        )


class MIRA_I_obs_with_scaled_Gaussian_noise(MiraPetriNetODESystem):
    """
    This is the same as the MIRA model, but with Scaled Gaussian noise instead of ScaledBeta noise.
    """

    def __init__(
        self,
        G: mira.modeling.Model,
        total_population: int = 1,
        data_reliability: float = 4.0,
    ):
        """
        Initialize the model with the total population and the data reliability.
        The more reliable the data, the smaller the noise variance.
        """
        self.total_population = total_population
        self.data_reliability = data_reliability
        for param_info in G.parameters.values():
            param_value = param_info.value
            if param_value is None:
                param_info.value = pyro.distributions.Uniform(0.0, 1.0)
            elif isinstance(param_value, (int, float)):
                param_info.value = pyro.distributions.Uniform(
                    max(0.9 * param_value, 0.0), 1.1 * param_value
                )
        super().__init__(G)

    @pyro.nn.pyro_method
    def observation_model(self, solution: Solution, var_name: str) -> None:
        """
        In the observation model, I_obs is the sum of I and Iv.
        We scale the noise by the square root of the mean,
        so that the noise variance is proportional to the mean.
        """
        if var_name == "I_obs":
            mean = solution["I"] + solution["I_v"]
        else:
            mean = solution[var_name]
        pyro.sample(
            var_name,
            dist.Normal(mean, torch.sqrt(mean) / self.data_reliability).to_event(1),
        )

    def __repr__(self):
        par_string = ",\n\t".join(
            [f"{get_name(p)} = {p.value}" for p in self.G.parameters.values()]
        )
        tp_string = f"total_population = {self.total_population}"
        dr_string = f"data_reliability = {self.data_reliability}"

        return f"{self.__class__.__name__}(\n\t{par_string},\n\t{tp_string},\n\t{dr_string}\n)"
