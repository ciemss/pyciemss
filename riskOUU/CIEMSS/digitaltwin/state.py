######## Written by: Anirban Chaudhuri (Oden Institute; UT Austin)
import numpy as np
from scipy.stats import truncnorm


class ODEState:
    def __init__(self, beta=None, betaV=None, gamma=None, gammaV=None, joint=None):
        # TODO: needs automated system to add parameters according to specified model
        self.beta = beta
        self.betaV = betaV
        self.gamma = gamma
        self.gammaV = gammaV
        # beta, betaV, gamma, gammaV = 0.2, 0.15, 1./10, 1.5/10, 0.005 
        # self.nu = nu
        # self.post_dist = joint # posterior distribution
        # TODO: additional random variables
        # Initial conditions: V0, I0, Iv0, R0


    # --------------------------
    @classmethod
    def setup_prior_fn(cls, mu, sigma, lb, ub):
        '''
        Setup prior function
        # TODO: currently only truncated normal distribution is supported
        '''
        return truncnorm((lb-mu)/sigma, (ub-mu)/sigma, loc=mu, scale=sigma)

    def prior(self, prior_info=None):
        """
        If called, assign priors as Digital State
        TODO: needs automated system to add parameters according to specified model
        """

        if prior_info is None:
            prior_info = {}
            prior_info["beta"] = {"mu": 0.2, "sigma": 0.025, "lb": 0.01, "ub": 0.3}
            prior_info["betaV"] = {"mu": 0.1, "sigma": 0.025, "lb": 0.01, "ub": 0.25}
            prior_info["gamma"] = {"mu": 0.1, "sigma": 0.1, "lb": 0.05, "ub": 0.4}
            prior_info["gammaV"] = {"mu": 0.2, "sigma": 0.1, "lb": 0.1, "ub": 0.4}

        self.beta = self.setup_prior_fn(prior_info["beta"]["mu"],
                                                    prior_info["beta"]["sigma"],
                                                    prior_info["beta"]["lb"],
                                                    prior_info["beta"]["ub"])

        self.betaV = self.setup_prior_fn(prior_info["betaV"]["mu"],
                                                    prior_info["betaV"]["sigma"],
                                                    prior_info["betaV"]["lb"],
                                                    prior_info["betaV"]["ub"])

        self.gamma = self.setup_prior_fn(prior_info["gamma"]["mu"],
                                                    prior_info["gamma"]["sigma"],
                                                    prior_info["gamma"]["lb"],
                                                    prior_info["gamma"]["ub"])

        self.gammaV = self.setup_prior_fn(prior_info["gammaV"]["mu"],
                                                    prior_info["gammaV"]["sigma"],
                                                    prior_info["gammaV"]["lb"],
                                                    prior_info["gammaV"]["ub"])

        return self

    
    def sample(self, num_samples=1, rseed=1):
        """
        Create a NumPy array of samples
        TODO: need to account for posterior distribution types
        """
        rv_list = [self.beta.rvs, self.betaV.rvs, self.gamma.rvs, self.gammaV.rvs]
        theta = np.stack([rv(size=num_samples, random_state=rseed) for rv in rv_list]).T

        return theta