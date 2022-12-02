import torch
import pyro
import pyro.distributions as dist

from pyro.nn import PyroModule, PyroSample, pyro_method

from pyciemss.ODE.abstract import ODE
from pyciemss.utils import state_flux_constraint, elvis


class SVIIvR(ODE):
    def __init__(self, 
                N,
                noise_prior=dist.Uniform(1., 10.),
                beta_prior=dist.Uniform(0.15, 0.25), 
                betaV_prior=dist.Uniform(0.05, 0.1),
                gamma_prior=dist.Uniform(0.05, 0.15),
                gammaV_prior=dist.Uniform(0.1, 0.2),
                nu_prior=dist.Uniform(0.02, 0.05),
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
    def deriv(self, t, state):
        S, V, I, Iv, R = state

        # Local fluxes exposed to pyro for interventions.
        # Note: This only works with solvers that use fixed time increments, such as Euler's method. Otherwise, we have name collisions.
        # positive/negative constraints ensure that we don't have vaccinated people become susceptible, etc.
        SV_flux_  = pyro.deterministic("SV_flux %f" % (t),  -self.nu * S)
        SI_flux_  = pyro.deterministic("SI_flux %f" % (t),  -self.beta  * S * (I + Iv) / self.N)
        VIv_flux_ = pyro.deterministic("VIv_flux %f" % (t), -self.betaV * V * (I + Iv) / self.N)
        IR_flux_  = pyro.deterministic("IR_flux %f" % (t),  -self.gamma * I)
        IvR_flux_ = pyro.deterministic("IvR_flux %f" % (t), -self.gammaV * Iv)

        SV_flux = state_flux_constraint(S, elvis(self.SV_flux, SV_flux_))
        SI_flux = state_flux_constraint(S, elvis(self.SI_flux, SI_flux_))
        VIv_flux = state_flux_constraint(V, elvis(self.VIv_flux, VIv_flux_))
        IR_flux = state_flux_constraint(I, elvis(self.IR_flux, IR_flux_))
        IvR_flux = state_flux_constraint(Iv, elvis(self.IvR_flux, IvR_flux_))


        dSdt  = SI_flux + SV_flux
        dVdt  = VIv_flux - SV_flux
        dIdt  = -SI_flux + IR_flux
        dIvdt =  -VIv_flux + IvR_flux
        dRdt  = -IR_flux - IvR_flux
        
        return dSdt, dVdt, dIdt, dIvdt, dRdt

    @pyro_method
    def param_prior(self):

        self.noise_var = pyro.sample("noise_var", self.noise_prior)
        self.beta      = pyro.sample("beta", self.beta_prior)
        self.betaV     = pyro.sample("betaV", self.betaV_prior)
        self.gamma     = pyro.sample("gamma", self.gamma_prior)
        self.gammaV    = pyro.sample("gammaV", self.gammaV_prior)
        self.nu        = pyro.sample("nu", self.nu_prior)

        # Initalize global flux values deterministically as torch.nan. This is a shortcut for "apply a global intervention", as these values
        # overwrite the local time-dependent ones when they are not torch.nans, i.e. if and only if an intervention has been applied.
        self.SV_flux =  pyro.deterministic("SV_flux", torch.tensor([torch.nan]))
        self.SI_flux =  pyro.deterministic("SI_flux", torch.tensor([torch.nan]))
        self.VIv_flux = pyro.deterministic("VIv_flux", torch.tensor([torch.nan]))
        self.IR_flux =  pyro.deterministic("IR_flux", torch.tensor([torch.nan]))
        self.IvR_flux = pyro.deterministic("IvR_flux", torch.tensor([torch.nan]))

    @pyro_method
    def observation_model(self, solution, data):
        S, V, I, Iv, R = solution

        # It's a little clunky that we have to do `None` handling for each implementation of 'observation_model'...
        if data == None:
            data = {k: None for k in ["S_obs", "V_obs", "I_obs", "R_obs"]}

        S_obs = pyro.sample("S_obs", dist.Normal(S, self.noise_var).to_event(1), obs=data["S_obs"])
        V_obs = pyro.sample("V_obs", dist.Normal(V, self.noise_var).to_event(1), obs=data["V_obs"])
        # We only observe the total number of infected people we don't know which of them are vaccinated.
        I_obs = pyro.sample("I_obs", dist.Normal(I + Iv, self.noise_var).to_event(1), obs=data["I_obs"])
        R_obs = pyro.sample("R_obs", dist.Normal(R, self.noise_var).to_event(1), obs=data["R_obs"])

        return S_obs, V_obs, I_obs, R_obs