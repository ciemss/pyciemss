import torch
import pyro
import pyro.distributions as dist

from pyro.nn import PyroModule, PyroSample, pyro_method

from pyciemss.ODE.abstract import ODE


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
        
    def deriv(self, t, state):
        S, V, I, Iv, R = state
        dSdt  = -self.beta  * S * I  / self.N - self.beta  * S * Iv / self.N - self.nu * S 
        dVdt  = -self.betaV * V * Iv / self.N - self.betaV * V * I  / self.N + self.nu * S
        dIdt  =  self.beta  * S * I  / self.N + self.beta  * S * Iv / self.N - self.gamma  * I 
        dIvdt = self.betaV * V * I  / self.N + self.betaV * V * Iv / self.N - self.gammaV * Iv 
        dRdt  =  self.gamma * I + self.gammaV * Iv

        return dSdt, dVdt, dIdt, dIvdt, dRdt

    @pyro_method
    def param_prior(self):
        self.noise_var = pyro.sample("noise_var", self.noise_prior)
        self.beta      = pyro.sample("beta", self.beta_prior)
        self.betaV     = pyro.sample("betaV", self.betaV_prior)
        self.gamma     = pyro.sample("gamma", self.gamma_prior)
        self.gammaV    = pyro.sample("gammaV", self.gammaV_prior)
        self.nu        = pyro.sample("nu", self.nu_prior)


    @pyro_method
    def observation_model(self, solution, data):
        S_, V_, I_, Iv_, R_ = solution

        # It's a little clunky that we have to do `None` handling for each implementation of 'observation_model'...
        # TODO: abstract out.
        
        if data == None:
            data = {k: None for k in ["S_obs", "V_obs", "I_obs", "Iv_obs", "R_obs"]}

        S  = pyro.sample("S_obs", dist.Normal(S_, self.noise_var).to_event(1), obs=data["S_obs"])
        V  = pyro.sample("V_obs", dist.Normal(V_, self.noise_var).to_event(1), obs=data["V_obs"])
        I  = pyro.sample("I_obs", dist.Normal(I_, self.noise_var).to_event(1), obs=data["I_obs"])
        Iv = pyro.sample("Iv_obs", dist.Normal(Iv_, self.noise_var).to_event(1), obs=data["Iv_obs"])
        R  = pyro.sample("R_obs", dist.Normal(R_, self.noise_var).to_event(1), obs=data["R_obs"])

        return S, V, I, Iv, R