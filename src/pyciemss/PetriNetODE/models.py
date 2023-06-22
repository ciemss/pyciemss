from typing import Dict, Optional
import functools
import operator
import torch
import pyro
import pyro.distributions as dist

from pyro.nn import pyro_method
import mira
from pyciemss.PetriNetODE.base import (
    ScaledBetaNoisePetriNetODESystem,
    MiraPetriNetODESystem,
    PetriNetODESystem,
    Time,
    State,
    Solution,
    get_name,
)
from pyciemss.utils import state_flux_constraint
from pyciemss.utils.distributions import ScaledBeta


class MiraRegNetODESystem(ScaledBetaNoisePetriNetODESystem):
    """
    MIRA RegNet model.
    """
    def deriv(self, t: Time, state: State) -> State:
        """compute the state derivative at time t.
        :param t: time
        :param state: state vector
        :return: state derivative vector
        """
        states = {k: state[i] for i, k in enumerate(self.var_order.values())}
        derivs = {k: 0. for k in states}

        population_size = sum(states.values())

        for transition in self.G.transitions.values():
            flux = getattr(self, get_name(transition.rate)) * functools.reduce(
                operator.mul, [states[k] for k in transition.consumed], 1
            )
            if len(transition.control) > 0:
                flux = flux * functools.reduce(operator.mul, [states[k] for k in transition.control], 1) / population_size**len(transition.control)

            for c in transition.consumed:
                derivs[c] -= flux
            for p in transition.produced:
                derivs[p] += flux

        return tuple(derivs[v] for v in self.var_order.values())

    
    

class LotkaVolterra(PetriNetODESystem):
    """Lotka-Volterra model built by hand to compare against MIRA Regnet model.
    See https://github.com/ciemss/pyciemss/issues/153
    for more detail.
    """
    def __init__(self, alpha: float, beta: float,
                 gamma: float, delta: float,
                 add_uncertainty=True,
                 pseudocount=1) -> None:
        """initialize alpha, beta, gamma, delta priors"""
        self.add_uncertainty = add_uncertainty
        self.pseudocount = pseudocount
        super().__init__()
        if self.add_uncertainty:
            self.alpha_prior = pyro.distributions.Uniform(max(0.9 * alpha, 0.0), 1.1 * alpha)
            self.beta_prior = pyro.distributions.Uniform(max(0.9 * beta, 0.0), 1.1 * beta)
            self.gamma_prior = pyro.distributions.Uniform(max(0.9 * gamma, 0.0), 1.1 * gamma)
            self.delta_prior = pyro.distributions.Uniform(max(0.9 * delta, 0.0), 1.1 * delta)
            self.gamma_prior = pyro.distributions.Uniform(max(0.9 * gamma, 0.0), 1.1 * gamma)
        else:
            self.alpha_prior = torch.nn.Parameter(torch.as_tensor(alpha))
            self.beta_prior = torch.nn.Parameter(torch.as_tensor(beta))
            self.gamma_prior = torch.nn.Parameter(torch.as_tensor(gamma))
            self.delta_prior = torch.nn.Parameter(torch.as_tensor(delta))


    @pyro.nn.pyro_method
    def param_prior(self) -> None:
        '''
        Inplace method defining the prior distribution over model parameters.
        All random variables must be defined using `pyro.sample` or `PyroSample` methods.
        '''
        if self.add_uncertainty:
            self.alpha = pyro.sample('alpha', self.alpha_prior)
            self.beta = pyro.sample('beta', self.beta_prior)
            self.gamma = pyro.sample('gamma', self.gamma_prior)
            self.delta = pyro.sample('delta', self.delta_prior)
        else:
            self.alpha = pyro.param('alpha', self.alpha_prior)
            self.beta = pyro.param('beta', self.beta_prior)
            self.gamma = pyro.param('gamma', self.gamma_prior)
            self.delta = pyro.param('delta', self.delta_prior)

    @pyro.nn.pyro_method
    def deriv(self, t: Time, state: State) -> State:
        """compute the state derivative at time t
        :param t: time
        :param state: state vector
        :return: state derivative vector
        """
        prey_population, predator_population = state
        dxdt = self.alpha * prey_population - self.beta * prey_population * predator_population
        dydt = self.delta * prey_population * predator_population - self.gamma * predator_population
        return dxdt, dydt
              
    def create_var_order(self) -> dict[str, int]:
        """create the variable order for the state vector"""
        return dict(prey_population=0, predator_population=1)

    @pyro.nn.pyro_method
    def observation_model(self, solution: Solution, var_name: str) -> None:
        """In the observation model, I_obs is the sum of I and Iv.  We scale the noise by the square root of the mean, so that the noise variance is proportional to the mean."""
        mean = solution[var_name]
        pyro.sample(
            var_name,
            dist.Normal(mean, torch.sqrt(mean) / self.pseudocount).to_event(1),
        )

    # @pyro.nn.pyro_method
    # def observation_model(self, solution: Solution, var_name: str) -> None:
    #     """define the observation model for the given variable
    #     :param solution: solution of the ODE system
    #     :param var_name: variable name
    #     """
    #     mean = solution[var_name]
    #     pseudocount = self.pseudocount
    #     total_population = sum(solution.values())
    #     pyro.sample(var_name, ScaledBeta(mean, total_population, pseudocount).to_event(1))

    def static_parameter_intervention(self, parameter: str, value: torch.Tensor) -> None:
        """set a static parameter intervention
        :param parameter: parameter name
        :param value: parameter value
        """
        setattr(self, parameter, value)


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
        self.beta_prior =  pyro.distributions.Uniform(max(0.9 * beta, 0.0), 1.1 * beta)
        self.gamma_prior = pyro.distributions.Uniform(max(0.9 * gamma, 0.0), 1.1 * gamma)
        self.pseudocount = pseudocount


    def create_var_order(self) -> dict[str, int]:
        """create the variable order for the state vector"""
        return {"susceptible_population": 0, "infected_population": 1, "immune_population": 2}

    @pyro.nn.pyro_method
    def deriv(self, t: Time, state: State) -> State:
        """compute the state derivative at time t
        :param t: time
        :param state: state vector
        :return: state derivative vector
        """
        assert torch.isclose(sum(state),self.total_population),f"The sum of state variables {state} is not scaled to the total population {self.total_population}."
        S, I, R = state
        dSdt = -self.beta * S * I / self.total_population
        dIdt = self.beta * S * I / self.total_population - self.gamma * I
        dRdt = self.gamma * I
        return dSdt, dIdt, dRdt

    @pyro.nn.pyro_method
    def param_prior(self) -> None:
        """define the prior distributions for the parameters"""
        self.beta =  pyro.sample('beta', self.beta_prior)
        self.gamma = pyro.sample('gamma', self.gamma_prior)

    @pyro.nn.pyro_method
    def observation_model(self, solution: Solution, var_name: str) -> None:
        """define the observation model for the given variable
        :param solution: solution of the ODE system
        :param var_name: variable name
        """
        mean = solution[var_name]
        pseudocount = self.pseudocount
        pyro.sample(var_name, ScaledBeta(mean, self.total_population, pseudocount).to_event(1))

    def static_parameter_intervention(self, parameter: str, value: torch.Tensor) -> None:
        """set a static parameter intervention
        :param parameter: parameter name
        :param value: parameter value
        """
        setattr(self, parameter, value)

class SVIIvR(PetriNetODESystem):
    def __init__(
        self,
        N,
        noise_prior=dist.Uniform(5.0, 10.0),
        beta_prior=dist.Uniform(0.1, 0.3),
        betaV_prior=dist.Uniform(0.025, 0.05),
        gamma_prior=dist.Uniform(0.05, 0.35),
        gammaV_prior=dist.Uniform(0.1, 0.4),
        nu_prior=dist.Uniform(0.001, 0.01),
    ):
        super().__init__()

        self.N = N
        self.noise_prior = noise_prior
        self.beta_prior = beta_prior
        self.betaV_prior = betaV_prior
        self.gamma_prior = gamma_prior
        self.gammaV_prior = gammaV_prior
        self.nu_prior = nu_prior

    ### TODO: write a hand-coded version of SVIIvR using the new PetriNetODESystem class.
    # @pyro_method
    # def deriv(self, t: Time, state: State) -> State:
    #     S, V, I, Iv, R = state

    #     # Local fluxes exposed to pyro for interventions.
    #     # Note: This only works with solvers that use fixed time increments, such as Euler's method. Otherwise, we have name collisions.
    #     SV_flux_ = pyro.deterministic("SV_flux %f" % (t), self.nu * S)
    #     SI_flux_ = pyro.deterministic(
    #         "SI_flux %f" % (t), self.beta * S * (I + Iv) / self.N
    #     )
    #     VIv_flux_ = pyro.deterministic(
    #         "VIv_flux %f" % (t), self.betaV * V * (I + Iv) / self.N
    #     )
    #     IR_flux_ = pyro.deterministic("IR_flux %f" % (t), self.gamma * I)
    #     IvR_flux_ = pyro.deterministic("IvR_flux %f" % (t), self.gammaV * Iv)

    #     # these state_flux_constraints ensure that we don't have vaccinated people become susceptible, etc.
    #     SV_flux = state_flux_constraint(S, SV_flux_)
    #     SI_flux = state_flux_constraint(S, SI_flux_)
    #     VIv_flux = state_flux_constraint(V, VIv_flux_)
    #     IR_flux = state_flux_constraint(I, IR_flux_)
    #     IvR_flux = state_flux_constraint(Iv, IvR_flux_)

    #     # Where the real magic happens.
    #     dSdt = -SI_flux - SV_flux
    #     dVdt = -VIv_flux + SV_flux
    #     dIdt = SI_flux - IR_flux
    #     dIvdt = VIv_flux - IvR_flux
    #     dRdt = IR_flux + IvR_flux

    #     return dSdt, dVdt, dIdt, dIvdt, dRdt

    @pyro_method
    def param_prior(self) -> None:

        self.noise_var = pyro.sample("noise_var", self.noise_prior)
        self.beta = pyro.sample("beta", self.beta_prior)
        self.betaV = pyro.sample("betaV", self.betaV_prior)
        self.gamma = pyro.sample("gamma", self.gamma_prior)
        self.gammaV = pyro.sample("gammaV", self.gammaV_prior)
        self.nu = pyro.sample("nu", self.nu_prior)

    @pyro_method
    def observation_model(
        self, solution: Solution, data: Optional[Dict[str, State]] = None
    ) -> Solution:
        S, V, I, Iv, R = solution

        # It's a little clunky that we have to do `None` handling for each implementation of 'observation_model'...
        if data == None:
            data = {k: None for k in ["S_obs", "V_obs", "I_obs", "R_obs"]}

        # TODO: Make sure observations are strictly greater than 0.

        S_obs = pyro.sample(
            "S_obs", dist.Normal(S, self.noise_var).to_event(1), obs=data["S_obs"]
        )
        V_obs = pyro.sample(
            "V_obs", dist.Normal(V, self.noise_var).to_event(1), obs=data["V_obs"]
        )
        # We only observe the total number of infected people we don't know which of them are vaccinated.
        I_obs = pyro.sample(
            "I_obs", dist.Normal(I + Iv, self.noise_var).to_event(1), obs=data["I_obs"]
        )
        R_obs = pyro.sample(
            "R_obs", dist.Normal(R, self.noise_var).to_event(1), obs=data["R_obs"]
        )

        return (S_obs, V_obs, I_obs, R_obs)


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
        """Initialize the model with the total population and the data reliability. The more reliable the data, the smaller the noise variance."""
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
        """In the observation model, I_obs is the sum of I and Iv.  We scale the noise by the square root of the mean, so that the noise variance is proportional to the mean."""
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
        return f"{self.__class__.__name__}(\n\t{par_string},\n\ttotal_population = {self.total_population},\n\tdata_reliability = {self.data_reliability}\n)"
