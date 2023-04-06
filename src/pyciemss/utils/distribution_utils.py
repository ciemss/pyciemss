import torch
import pyro.distributions as dist

def LogNormalPrior(mean, variance):
    mu = torch.log(mean**2/torch.sqrt((mean**2 + variance)))
    scale = torch.sqrt(torch.log(1 + variance/mean**2))
    return dist.LogNormal(mu, scale)

def LogNormalPrior_propVar(mean, cov=0.05):
    variance = mean*cov    
    return LogNormalPrior(mean, variance)