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
                 alpha_prior=dist.LogNormal(torch.log(torch.tensor(0.570)), torch.tensor(0.5)),
                 beta_prior=dist.LogNormal(torch.log(torch.tensor(0.011)), torch.tensor(0.5)),
                 gamma_prior=dist.LogNormal(torch.log(torch.tensor(0.0456)), torch.tensor(0.5)),
                 delta_prior=dist.LogNormal(torch.log(torch.tensor(0.011)), torch.tensor(0.5)),
                 epsilon_prior=dist.LogNormal(torch.log(torch.tensor(0.171)), torch.tensor(0.5)),
                 lamb_prior =dist.LogNormal(torch.log(torch.tensor(0.034)), torch.tensor(0.5)),
                 zeta_prior=dist.LogNormal(torch.log(torch.tensor(0.125)), torch.tensor(0.5)),
                 eta_prior=dist.LogNormal(torch.log(torch.tensor(0.125)), torch.tensor(0.5)),
                 kappa_prior=dist.LogNormal(torch.log(torch.tensor(0.017)), torch.tensor(0.5)),
                 theta_prior=dist.LogNormal(torch.log(torch.tensor(0.371)), torch.tensor(0.5)),
                 rho_prior=dist.LogNormal(torch.log(torch.tensor(0.034)), torch.tensor(0.5)),
                 xi_prior=dist.LogNormal(torch.log(torch.tensor(0.017)), torch.tensor(0.5)),
                 sigma_prior=dist.LogNormal(torch.log(torch.tensor(0.017)), torch.tensor(0.5)),
                 mu_prior=dist.LogNormal(torch.log(torch.tensor(0.017)), torch.tensor(0.5)),
                 nu_prior=dist.LogNormal(torch.log(torch.tensor(0.027)), torch.tensor(0.5)),
                 tau_prior=dist.LogNormal(torch.log(torch.tensor(0.01)), torch.tensor(0.5)),
                ):
        super().__init__()

        self.N = N
        self.alpha_prior= alpha_prior
        self.beta_prior= beta_prior
        self.gamma_prior= gamma_prior
        self.delta_prior= delta_prior
        self.epsilon_prior = epsilon_prior
        self.lamb_prior  = lamb_prior
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
        SI_flux_  = pyro.deterministic("SI_flux %f" % (t),  self.alpha * S * I / self.N)
        SD_flux_  = pyro.deterministic("SD_flux %f" % (t),  self.beta  * S * D / self.N)
        SA_flux_ =  pyro.deterministic("SA_flux %f" % (t),  self.gamma * S * A / self.N)
        SR_flux_ =  pyro.deterministic("SR_flux %f" % (t),  self.delta * S * R / self.N)

        ID_flux_  = pyro.deterministic("ID_flux %f" % (t),  self.epsilon * I)
        IH_flux_  = pyro.deterministic("IH_flux %f" % (t),  self.lamb    * I)
        IA_flux_  = pyro.deterministic("IA_flux %f" % (t),  self.zeta    * I)
        DR_flux_  = pyro.deterministic("DR_flux %f" % (t),  self.eta     * D)
        AH_flux_  = pyro.deterministic("AH_flux %f" % (t),  self.kappa   * A)
        AR_flux_  = pyro.deterministic("AR_flux %f" % (t),  self.theta   * A)
        DH_flux_  = pyro.deterministic("DH_flux %f" % (t),  self.rho     * D)
        RH_flux_  = pyro.deterministic("RH_flux %f" % (t),  self.xi      * R)
        TH_flux_  = pyro.deterministic("TH_flux %f" % (t),  self.sigma   * T)
        AT_flux_  = pyro.deterministic("AT_flux %f" % (t),  self.mu      * A)
        RT_flux_  = pyro.deterministic("RT_flux %f" % (t),  self.nu      * R)
        TE_flux_  = pyro.deterministic("TE_flux %f" % (t),  self.tau     * T)


        # these state_flux_constraints ensure that we don't have vaccinated people become susceptible, etc.
        SI_flux  = state_flux_constraint(S, SI_flux_)
        SD_flux  = state_flux_constraint(S, SD_flux_)
        SA_flux =  state_flux_constraint(S, SA_flux_)
        SR_flux =  state_flux_constraint(S, SR_flux_)
        ID_flux  = state_flux_constraint(I, ID_flux_)
        IH_flux  = state_flux_constraint(I, IH_flux_)
        IA_flux  = state_flux_constraint(I, IA_flux_)
        DR_flux  = state_flux_constraint(D, DR_flux_)
        AH_flux  = state_flux_constraint(A, AH_flux_)
        AR_flux  = state_flux_constraint(A, AR_flux_)
        DH_flux  = state_flux_constraint(D, DH_flux_)
        RH_flux  = state_flux_constraint(R, RH_flux_)
        TH_flux  = state_flux_constraint(T, TH_flux_)
        AT_flux  = state_flux_constraint(A, AH_flux_)
        RT_flux  = state_flux_constraint(R, RH_flux_)
        TE_flux  = state_flux_constraint(T, TE_flux_)

        # Where the real magic happens.
        dSdt  = -SI_flux - SD_flux - SA_flux - SR_flux
        dIdt  = SI_flux + SD_flux + SA_flux + SR_flux - ID_flux - IH_flux - IA_flux
        dDdt  = ID_flux - DR_flux - DH_flux
        dAdt  = IA_flux - AH_flux - AR_flux - AT_flux
        dRdt  = DR_flux + AR_flux - RH_flux - RT_flux
        dTdt  = AT_flux - TH_flux - TE_flux
        dHdt  = IH_flux + DH_flux + AH_flux + RH_flux + TH_flux
        dEdt  = TE_flux

        return dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt

    @pyro_method
    def param_prior(self) -> None:
        self.alpha = pyro.sample("alpha", self. alpha_prior)
        self.beta = pyro.sample("beta", self. beta_prior)
        self.gamma = pyro.sample("gamma", self. gamma_prior)
        self.delta = pyro.sample("delta", self. delta_prior)
        self.epsilon  = pyro.sample("epsilon", self. epsilon_prior)
        self.lamb   = pyro.sample("lamb", self. lamb_prior)
        self.zeta  = pyro.sample("zeta", self. zeta_prior)
        self.eta  = pyro.sample("eta", self. eta_prior)
        self.kappa  = pyro.sample("kappa", self. kappa_prior)
        self.theta  = pyro.sample("theta", self. theta_prior)
        self.rho  = pyro.sample("rho", self. rho_prior)
        self.xi  = pyro.sample("xi", self. xi_prior)
        self.sigma  = pyro.sample("sigma", self. sigma_prior)
        self.mu  = pyro.sample("mu", self. mu_prior)
        self.nu  = pyro.sample("nu", self. nu_prior)
        self.tau = pyro.sample("tau", self.tau_prior)


    def observation_model(self, solution: Solution, data: Optional[Dict[str, State]] = None) -> Solution:
        S, I, D, A, R, T, H, E = solution
        # It's a little clunky that we have to do `None` handling for each implementation of 'observation_model'...
        if data == None:
            data = {k: None for k in ["S_obs", "I_obs", "D_obs", "A_obs", "R_obs", "T_obs", "H_obs", "E_obs"]}

        # TODO: Make sure observations are strictly greater than 0.

        S_obs = pyro.deterministic("S_obs", S)
        I_obs = pyro.deterministic("I_obs", I)
        D_obs = pyro.deterministic("D_obs", D)
        A_obs = pyro.deterministic("A_obs", A)
        R_obs = pyro.deterministic("R_obs", R)
        T_obs = pyro.deterministic("T_obs", T)
        H_obs = pyro.deterministic("H_obs", H)
        E_obs = pyro.deterministic("E_obs", E)

        return (S_obs, I_obs, D_obs, A_obs,  R_obs, T_obs, H_obs, E_obs)
