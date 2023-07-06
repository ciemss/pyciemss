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


class SEIARHD(PetriNetODESystem):
    """Susceptible (S), Exposed(E), Symptomatic Infectious (I), Asymptomatic Infectious (A), Recovered (R), Hospitalized (H), Deceased (D)."""
    def __init__(
            self,
            N: int,
            beta: float,
            delta: float,
            alpha: float,
            pS: float,
            gamma: float,
            hosp: float,
            los: float,
            dh: float,
            dnh: float,
            pseudocount: float = 1.0,
            ) -> None:
        """initialize parameters
        :param N: total population
        :param beta: transmission rate
        :param delta: difference in infectiousness symptomatic/asymptomatic
        :param alpha: latency period
        :param pS: percent of exposures which become symptomatic
        :param gamma: recovery rate
        :param hosp: hospitalization rate of infectious individuals
        :param los: average length (days) of hospital stay
        :param dh: death rate of hospitalized individuals
        :param dnh: death rate of infectious individuals (never hospitalized)
        """
        super().__init__()
        self.total_population = N
        self.beta_prior =  pyro.distributions.Uniform(max(0.9 * beta, 0.0), 1.1 * beta)
        self.delta_prior = pyro.distributions.Uniform(max(0.9 * delta, 0.0), 1.1 * delta)
        self.alpha_prior = pyro.distributions.Uniform(max(0.9 * alpha, 0.0), 1.1 * alpha)
        self.pS_prior = pyro.distributions.Uniform(max(0.9 * pS, 0.0), 1.1 * pS)
        self.gamma_prior = pyro.distributions.Uniform(max(0.9 * gamma, 0.0), 1.1 * gamma)
        self.hosp_prior = pyro.distributions.Uniform(max(0.9 * hosp, 0.0), 1.1 * hosp)
        self.los_prior = pyro.distributions.Uniform(max(0.9 * los, 0.0), 1.1 * los)
        self.dh_prior = pyro.distributions.Uniform(max(0.9 * dh, 0.0), 1.1 * dh)
        self.dnh_prior = pyro.distributions.Uniform(max(0.9 * dnh, 0.0), 1.1 * dnh)
        self.pseudocount = pseudocount


    def create_var_order(self) -> dict[str, int]:
        """create the variable order for the state vector"""
        return {"susceptible_population": 0, "exposed_population": 1, "symptomatic_population": 2, 
                "asymptomatic_population": 3, "recovered_population": 4, 
                "hospitalized_population": 5, "deceased_population": 6}

    @pyro.nn.pyro_method
    def deriv(self, t: Time, state: State) -> State:
        """compute the state derivative at time t
        :param t: time
        :param state: state vector
        :return: state derivative vector
        """
        assert torch.isclose(sum(state),self.total_population),f"The sum of state variables {state} is not scaled to the total population {self.total_population}."
        S, E, I, A, R, H, D = state
        dSdt = -self.beta * S * (self.delta * I + A) / self.total_population
        dEdt = self.beta * S * (self.delta * I + A) / self.total_population - (1/self.alpha) * E
        dIdt = (self.pS/self.alpha) * E - self.gamma * I
        dAdt = ((1 - self.pS)/self.alpha) * E - self.gamma * A
        dRdt = self.gamma * (1 - self.hosp - self.dnh) * I + self.gamma * A + ((1 - self.dh)/self.los) * H
        dHdt = self.gamma * self.hosp * I - (1/self.los) * H
        dDdt = self.gamma * self.dnh * I + (self.dh/self.los) * H
        return dSdt, dEdt, dIdt, dAdt, dRdt, dHdt, dDdt

    @pyro.nn.pyro_method
    def param_prior(self) -> None:
        """define the prior distributions for the parameters"""
        setattr(self, 'beta', pyro.sample('beta', self.beta_prior))
        setattr(self, 'delta', pyro.sample('delta', self.delta_prior))
        setattr(self, 'alpha', pyro.sample('alpha', self.alpha_prior))
        setattr(self, 'pS', pyro.sample('pS', self.pS_prior))
        setattr(self, 'gamma', pyro.sample('gamma', self.gamma_prior))
        setattr(self, 'hosp', pyro.sample('hosp', self.hosp_prior))
        setattr(self, 'los', pyro.sample('los', self.los_prior))
        setattr(self, 'dh', pyro.sample('dh', self.dh_prior))
        setattr(self, 'dnh', pyro.sample('dnh', self.dnh_prior))

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

    def create_start_state_symp(self, total_population):
        """Create a start state with a single symptomatic individual."""
        returned_state = {}
        returned_state["exposed_population"] = 0
        returned_state["symptomatic_population"] = 1
        returned_state["asymptomatic_population"] = 0
        returned_state["recovered_population"] = 0
        returned_state["hospitalized_population"] = 0
        returned_state["deceased_population"] = 0
        returned_state["susceptible_population"] = total_population - sum(returned_state.values())
    
        assert(returned_state["susceptible_population"] > 0)
        return {k:v/total_population for k, v in returned_state.items()}

class SIRHD(PetriNetODESystem):
    # Susceptible (S), Infectious (I), Recovered (R), Hospitalized (H), Deceased (D)
    def __init__(
            self,
            N: int,
            beta: float,
            gamma: float,
            hosp: float,
            los: float,
            dh: float,
            dnh: float,
            pseudocount: float = 1.0,
            ) -> None:
        """initialize parameters
        :param N: total population
        :param beta: infection rate
        :param gamma: recovery rate
        :param hosp: hospitalization rate of infectious individuals
        :param los: average length (days) of hospital stay
        :param dh: death rate of hospitalized individuals
        :param dnh: death rate of infectious individuals (never hospitalized)
        """
        super().__init__()
        self.total_population = N
        self.beta_prior =  pyro.distributions.Uniform(max(0.9 * beta, 0.0), 1.1 * beta)
        self.gamma_prior = pyro.distributions.Uniform(max(0.9 * gamma, 0.0), 1.1 * gamma)
        self.hosp_prior = pyro.distributions.Uniform(max(0.9 * hosp, 0.0), 1.1 * hosp)
        self.los_prior = pyro.distributions.Uniform(max(0.9 * los, 0.0), 1.1 * los)
        self.dh_prior = pyro.distributions.Uniform(max(0.9 * dh, 0.0), 1.1 * dh)
        self.dnh_prior = pyro.distributions.Uniform(max(0.9 * dnh, 0.0), 1.1 * dnh)
        self.pseudocount = pseudocount


    def create_var_order(self) -> dict[str, int]:
        """create the variable order for the state vector"""
        return {"susceptible_population": 0, "infectious_population": 1, "recovered_population": 2, 
                "hospitalized_population": 3, "deceased_population": 4}

    @pyro.nn.pyro_method
    def deriv(self, t: Time, state: State) -> State:
        """compute the state derivative at time t
        :param t: time
        :param state: state vector
        :return: state derivative vector
        """
        assert torch.isclose(sum(state),self.total_population),f"The sum of state variables {state} is not scaled to the total population {self.total_population}."
        S, I, R, H, D = state
        dSdt = -self.beta * S * I / self.total_population
        dIdt = self.beta * S * I / self.total_population - self.gamma * I
        dRdt = self.gamma * (1 - self.hosp - self.dnh) * I + ((1 - self.dh)/self.los) * H
        dHdt = self.gamma * self.hosp * I - (1/self.los) * H
        dDdt = self.gamma * self.dnh * I + (self.dh/self.los) * H
        return dSdt, dIdt, dRdt, dHdt, dDdt

    @pyro.nn.pyro_method
    def param_prior(self) -> None:
        """define the prior distributions for the parameters"""
        setattr(self, 'beta', pyro.sample('beta', self.beta_prior))
        setattr(self, 'gamma', pyro.sample('gamma', self.gamma_prior))
        setattr(self, 'hosp', pyro.sample('hosp', self.hosp_prior))
        setattr(self, 'los', pyro.sample('los', self.los_prior))
        setattr(self, 'dh', pyro.sample('dh', self.dh_prior))
        setattr(self, 'dnh', pyro.sample('dnh', self.dnh_prior))

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

    def create_start_state_inf(self, total_population):
        """Create a start state with a single infectious individual."""
        returned_state = {}
        returned_state["infectious_population"] = 1
        returned_state["recovered_population"] = 0
        returned_state["hospitalized_population"] = 0
        returned_state["deceased_population"] = 0
        returned_state["susceptible_population"] = total_population - sum(returned_state.values())
    
        assert(returned_state["susceptible_population"] > 0)
        return {k:v/total_population for k, v in returned_state.items()}

class SEIARHDS(PetriNetODESystem):
    """ Susceptible (S), Exposed(E), Symptomatic Infectious (I), Asymptomatic Infectious (A), Recovered (R), Hospitalized (H), Deceased (D)."""
    def __init__(
            self,
            N: int,
            beta: float,
            delta: float,
            tau: float,
            alpha: float,
            pS: float,
            gamma: float,
            hosp: float,
            los: float,
            dh: float,
            dnh: float,
            pseudocount: float = 1.0,
            ) -> None:
        """initialize parameters
        :param N: total population
        :param beta: transmission rate
        :param delta: difference in infectiousness symptomatic/asymptomatic
        :param tau: immunity period
        :param alpha: latency period
        :param pS: percent of exposures which become symptomatic
        :param gamma: recovery rate
        :param hosp: hospitalization rate of infectious individuals
        :param los: average length (days) of hospital stay
        :param dh: death rate of hospitalized individuals
        :param dnh: death rate of infectious individuals (never hospitalized)
        """
        super().__init__()
        self.total_population = N
        self.beta_prior =  pyro.distributions.Uniform(max(0.9 * beta, 0.0), 1.1 * beta)
        self.delta_prior = pyro.distributions.Uniform(max(0.9 * delta, 0.0), 1.1 * delta)
        self.tau_prior = pyro.distributions.Uniform(max(0.9 * tau, 0.0), 1.1 * tau)
        self.alpha_prior = pyro.distributions.Uniform(max(0.9 * alpha, 0.0), 1.1 * alpha)
        self.pS_prior = pyro.distributions.Uniform(max(0.9 * pS, 0.0), 1.1 * pS)
        self.gamma_prior = pyro.distributions.Uniform(max(0.9 * gamma, 0.0), 1.1 * gamma)
        self.hosp_prior = pyro.distributions.Uniform(max(0.9 * hosp, 0.0), 1.1 * hosp)
        self.los_prior = pyro.distributions.Uniform(max(0.9 * los, 0.0), 1.1 * los)
        self.dh_prior = pyro.distributions.Uniform(max(0.9 * dh, 0.0), 1.1 * dh)
        self.dnh_prior = pyro.distributions.Uniform(max(0.9 * dnh, 0.0), 1.1 * dnh)
        self.pseudocount = pseudocount


    def create_var_order(self) -> dict[str, int]:
        """create the variable order for the state vector"""
        return {"susceptible_population": 0, "exposed_population": 1, "symptomatic_population": 2, 
                "asymptomatic_population": 3, "recovered_population": 4, 
                "hospitalized_population": 5, "deceased_population": 6}

    @pyro.nn.pyro_method
    def deriv(self, t: Time, state: State) -> State:
        """compute the state derivative at time t
        :param t: time
        :param state: state vector
        :return: state derivative vector
        """
        assert torch.isclose(sum(state),self.total_population),f"The sum of state variables {state} is not scaled to the total population {self.total_population}."
        S, E, I, A, R, H, D = state
        dSdt = -self.beta * S * (self.delta * I + A) / self.total_population + (1/self.tau) * R
        dEdt = self.beta * S * (self.delta * I + A) / self.total_population - (1/self.alpha) * E
        dIdt = (self.pS/self.alpha) * E - self.gamma * I
        dAdt = ((1 - self.pS)/self.alpha) * E - self.gamma * A
        dRdt = self.gamma * (1 - self.hosp - self.dnh) * I + self.gamma * A + ((1 - self.dh)/self.los) * H - (1/self.tau) * R
        dHdt = self.gamma * self.hosp * I - (1/self.los) * H
        dDdt = self.gamma * self.dnh * I + (self.dh/self.los) * H
        return dSdt, dEdt, dIdt, dAdt, dRdt, dHdt, dDdt

    @pyro.nn.pyro_method
    def param_prior(self) -> None:
        """define the prior distributions for the parameters"""
        setattr(self, 'beta', pyro.sample('beta', self.beta_prior))
        setattr(self, 'delta', pyro.sample('delta', self.delta_prior))
        setattr(self, 'tau', pyro.sample('tau', self.tau_prior))
        setattr(self, 'alpha', pyro.sample('alpha', self.alpha_prior))
        setattr(self, 'pS', pyro.sample('pS', self.pS_prior))
        setattr(self, 'gamma', pyro.sample('gamma', self.gamma_prior))
        setattr(self, 'hosp', pyro.sample('hosp', self.hosp_prior))
        setattr(self, 'los', pyro.sample('los', self.los_prior))
        setattr(self, 'dh', pyro.sample('dh', self.dh_prior))
        setattr(self, 'dnh', pyro.sample('dnh', self.dnh_prior))

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

    def create_start_state_symp(self, total_population):
        """Create a start state with 1 symptomatic individual and the rest susceptible."""
        returned_state = {}
        returned_state["exposed_population"] = 0
        returned_state["symptomatic_population"] = 1
        returned_state["asymptomatic_population"] = 0
        returned_state["recovered_population"] = 0
        returned_state["hospitalized_population"] = 0
        returned_state["deceased_population"] = 0
        returned_state["susceptible_population"] = total_population - sum(returned_state.values())
    
        assert(returned_state["susceptible_population"] > 0)
        return {k:v/total_population for k, v in returned_state.items()}
