from typing import Dict, Optional

import torch
import pyro
import pyro.distributions as dist

from pyro.nn import pyro_method

from pyciemss.PetriNetODE.base import MiraPetriNetODESystem, PetriNetODESystem, Time, State, Solution
from pyciemss.utils import state_flux_constraint


class SVIIvR(PetriNetODESystem):
    def __init__(self,
                N,
                noise_prior=dist.Uniform(5., 10.),
                beta_prior=dist.Uniform(0.1, 0.3),
                betaV_prior=dist.Uniform(0.025, 0.05),
                gamma_prior=dist.Uniform(0.05, 0.35),
                gammaV_prior=dist.Uniform(0.1, 0.4),
                nu_prior=dist.Uniform(0.001, 0.01)
                ):
        super().__init__()

        self.N = N
        self.noise_prior  = noise_prior
        self.beta_prior   = beta_prior
        self.betaV_prior  = betaV_prior
        self.gamma_prior  = gamma_prior
        self.gammaV_prior = gammaV_prior
        self.nu_prior     = nu_prior

    @pyro_method
    def deriv(self, t: Time, state: State) -> State:
        S, V, I, Iv, R = state

        # Local fluxes exposed to pyro for interventions.
        # Note: This only works with solvers that use fixed time increments, such as Euler's method. Otherwise, we have name collisions.
        SV_flux_  = pyro.deterministic("SV_flux %f" % (t),  self.nu * S)
        SI_flux_  = pyro.deterministic("SI_flux %f" % (t),  self.beta  * S * (I + Iv) / self.N)
        VIv_flux_ = pyro.deterministic("VIv_flux %f" % (t), self.betaV * V * (I + Iv) / self.N)
        IR_flux_  = pyro.deterministic("IR_flux %f" % (t),  self.gamma * I)
        IvR_flux_ = pyro.deterministic("IvR_flux %f" % (t), self.gammaV * Iv)

        # these state_flux_constraints ensure that we don't have vaccinated people become susceptible, etc.
        SV_flux = state_flux_constraint(S,  SV_flux_)
        SI_flux = state_flux_constraint(S,  SI_flux_)
        VIv_flux = state_flux_constraint(V,  VIv_flux_)
        IR_flux = state_flux_constraint(I, IR_flux_)
        IvR_flux = state_flux_constraint(Iv, IvR_flux_)

        # Where the real magic happens.
        dSdt  = -SI_flux - SV_flux
        dVdt  = -VIv_flux + SV_flux
        dIdt  = SI_flux - IR_flux
        dIvdt = VIv_flux - IvR_flux
        dRdt  = IR_flux + IvR_flux

        return dSdt, dVdt, dIdt, dIvdt, dRdt

    @pyro_method
    def param_prior(self) -> None:

        self.noise_var = pyro.sample("noise_var", self.noise_prior)
        self.beta      = pyro.sample("beta", self.beta_prior)
        self.betaV     = pyro.sample("betaV", self.betaV_prior)
        self.gamma     = pyro.sample("gamma", self.gamma_prior)
        self.gammaV    = pyro.sample("gammaV", self.gammaV_prior)
        self.nu        = pyro.sample("nu", self.nu_prior)

    @pyro_method
    def observation_model(self, solution: Solution, data: Optional[Dict[str, State]] = None) -> Solution:
        S, V, I, Iv, R = solution

        # It's a little clunky that we have to do `None` handling for each implementation of 'observation_model'...
        if data is None:
            data = {k: None for k in ["S_obs", "V_obs", "I_obs", "R_obs"]}

        # TODO: Make sure observations are strictly greater than 0.

        S_obs = pyro.sample("S_obs", dist.Normal(S, self.noise_var).to_event(1), obs=data["S_obs"])
        V_obs = pyro.sample("V_obs", dist.Normal(V, self.noise_var).to_event(1), obs=data["V_obs"])
        # We only observe the total number of infected people we don't know which of them are vaccinated.
        I_obs = pyro.sample("I_obs", dist.Normal(I + Iv, self.noise_var).to_event(1), obs=data["I_obs"])
        R_obs = pyro.sample("R_obs", dist.Normal(R, self.noise_var).to_event(1), obs=data["R_obs"])

        return (S_obs, V_obs, I_obs, R_obs)


class MIRA_SVIIvR(MiraPetriNetODESystem):

    def __init__(self, G, *, noise_var: float = 1):
        super().__init__(G)
        self.register_buffer("noise_var", torch.as_tensor(noise_var))

    @pyro.nn.pyro_method
    def observation_model(self, solution: Solution, var_name: str) -> None:
        """
        This is the observation model for the MIRA model.
        It is assumed that I_obs is the sum of I and I_v.
        """
        named_solution = dict(zip(self.var_order, solution))
        if var_name == "I_obs":
            value = named_solution["I"] + named_solution["I_v"]
        else:
            value = named_solution[var_name]
        pyro.sample(
            var_name,
            pyro.distributions.Normal(value, self.noise_var).to_event(1),
        )
        
