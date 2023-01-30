from typing import Dict, Optional

import torch
import pyro
import pyro.distributions as dist

from pyro.nn import pyro_method

from pyciemss.ODE.base import ODE, Time, State, Solution, Observation
from pyciemss.utils import state_flux_constraint

class SVIIvR(ODE):
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
        if data == None:
            data = {k: None for k in ["S_obs", "V_obs", "I_obs", "R_obs"]}

        # TODO: Make sure observations are strictly greater than 0.

        S_obs = pyro.sample("S_obs", dist.Normal(S, self.noise_var).to_event(1), obs=data["S_obs"])
        V_obs = pyro.sample("V_obs", dist.Normal(V, self.noise_var).to_event(1), obs=data["V_obs"])
        # We only observe the total number of infected people we don't know which of them are vaccinated.
        I_obs = pyro.sample("I_obs", dist.Normal(I + Iv, self.noise_var).to_event(1), obs=data["I_obs"])
        R_obs = pyro.sample("R_obs", dist.Normal(R, self.noise_var).to_event(1), obs=data["R_obs"])

        return (S_obs, V_obs, I_obs, R_obs)



class SIDARTHE(ODE):
    def __init__(self,
                N=1,
                noise_prior=dist.Uniform(5., 10.),
                alpha_prior=dist.Uniform(0.1, 0.3),
                beta_prior=dist.Uniform(0.025, 0.05),
                gamma_prior=dist.Uniform(0.05, 0.35),
                delta_prior=dist.Uniform(0.1, 0.4),
                 epsilon_prior=dist.Uniform(0.001, 0.01),
                 lambda_prior =dist.Uniform(0.001, 0.01),
                 zeta_prior=dist.Uniform(0.001, 0.01),
                 eta_prior=dist.Uniform(0.001, 0.01),
                 kappa_prior=dist.Uniform(0.001, 0.01),
                 theta_prior=dist.Uniform(0.001, 0.01),
                 rho_prior=dist.Uniform(0.001, 0.01),
                 xi_prior=dist.Uniform(0.001, 0.01),
                 sigma_prior=dist.Uniform(0.001, 0.01),
                 mu_prior=dist.Uniform(0.001, 0.01),
                 nu_prior=dist.Uniform(0.001, 0.01),
                 tau_prior=dist.Uniform(0.001, 0.01)
                ):
        super().__init__()

        self.N = N
        self.noise_prior=noise_prior
        self.alpha_prior= alpha_prior
        self.beta_prior= beta_prior
        self.gamma_prior= gamma_prior
        self.delta_prior= delta_prior
        self.epsilon_prior = epsilon_prior
        self.lambda_prior  = lambda_prior
        self.zeta_prior = zeta_prior
        self.eta_prior = eta_prior
        self.kappa_prior = kappa_prior
        self.theta_prior = theta_prior
        self.rho_prior = rho_prior
        self.xi_prior = xi_prior
        self.sigma_prior = sigma_prior
        self.mu_prior = mu_prior
        self.nu_prior = nu_prior
        self.tau_prior=tau_prior


    @pyro_method
    def deriv(self, t: Time, state: State) -> State:
        # S, susceptible (uninfected);
        # I, infected (asymptomatic or pauci-symptomatic infected, undetected);
        # D, diagnosed (asymptomatic infected, detected);
        # A,  ailing (symptomatic infected, undetected);
        # R, recognized (symptomatic infected, detected);
        # T, threatened (infected with life-threatening symptoms,  detected);
        # H, healed (recovered);
        # E, extinct (dead).
        S, I, D, A, R, T, H, E = state

        # Local fluxes exposed to pyro for interventions.
        # Note: This only works with solvers that use fixed time increments, such as Euler's method. Otherwise, we have name collisions.
        SI_flux_  = pyro.deterministic("SI_flux %f" % (t),  self.nu * S)


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
        if data == None:
            data = {k: None for k in ["S_obs", "V_obs", "I_obs", "R_obs"]}

        # TODO: Make sure observations are strictly greater than 0.

        S_obs = pyro.sample("S_obs", dist.Normal(S, self.noise_var).to_event(1), obs=data["S_obs"])
        V_obs = pyro.sample("V_obs", dist.Normal(V, self.noise_var).to_event(1), obs=data["V_obs"])
        # We only observe the total number of infected people we don't know which of them are vaccinated.
        I_obs = pyro.sample("I_obs", dist.Normal(I + Iv, self.noise_var).to_event(1), obs=data["I_obs"])
        R_obs = pyro.sample("R_obs", dist.Normal(R, self.noise_var).to_event(1), obs=data["R_obs"])

        return (S_obs, V_obs, I_obs, R_obs)
