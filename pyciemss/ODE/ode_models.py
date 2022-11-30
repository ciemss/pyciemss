import torch
import pyro.distributions as dist

from pyro.nn import PyroModule, PyroSample

from pyciemss.ODE.abstractions import ODEModel


class SVIIvR(ODEModel):
    def __init__(self, 
                    N, 
                    beta_prior=dist.Uniform(0.15, 0.25), 
                    betaV_prior=dist.Uniform(0.1, 0.2),
                    gamma_prior=dist.Uniform(0.05, 0.15),
                    gammaV_prior=dist.Uniform(0.1, 0.2),
                    nu_prior=dist.Uniform(0.02, 0.08),
                    **kwargs):
        super().__init__(**kwargs)

        self.N      = N
        self.beta   = PyroSample(beta_prior)
        self.betaV  = PyroSample(betaV_prior)
        self.gamma  = PyroSample(gamma_prior)
        self.gammaV = PyroSample(gammaV_prior)
        self.nu     = PyroSample(nu_prior)
        
    def prior_sample(self):
        self.beta
        self.betaV
        self.gamma
        self.gammaV
        self.nu

    def forward(self, t, state):
        S, V, I, Iv, R = state
        dSdt = -self.beta  * S * I  / self.N - self.beta   * S * Iv / self.N - self.nu * S 
        dVdt = -self.betaV * V * Iv / self.N - self.betaV  * V * I  / self.N + self.nu * S
        dIdt =  self.beta  * S * I  / self.N  + self.beta  * S * Iv / self.N - self.gamma  * I 
        dIvdt = self.betaV * V * I / self.N   + self.betaV * V * Iv / self.N - self.gammaV * Iv 
        dRdt =  self.gamma * I + self.gammaV * Iv

        return torch.tensor([dSdt, dVdt, dIdt, dIvdt, dRdt])