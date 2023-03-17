from .petri_utils import (seq_id_suffix,
           load_sim_result,
           load,
           draw_petri,
           natural_order,
           add_state_indicies,
           register_template,
           natural_conversion,
           natural_degradation,
           natural_order,
           controlled_conversion,
           grouped_controlled_conversion,
           deterministic,
           petri_to_ode,
           order_state,
           unorder_state,
           duplicate_petri_net,
           intervene_petri_net)
from .inference_utils import (get_tspan,
                              state_flux_constraint,
                              run_inference,
                              is_density_equal,
                              is_intervention_density_equal)
import pyro

def log_normal_transform(mu, sigma) -> pyro.distributions.LogNormal:
    """
    :param mu: the mean of the LogNormal distribution
    :param sigma: the standard deviation of the LogNormal distribution
    If we want the lognormal distribution to have a mean of mu and a standard deviation of sigma,
    then we need to make loc = log(mu**2/(mu**2 + sigma**2)) and scale = sqrt( log(1 + sigma**2/mu**2))
     according to wikipedia: https://en.wikipedia.org/wiki/Log-normal_distribution
    """
    return dist.LogNormal(loc = torch.log(mu**2/torch.sqrt(sigma**2 + mu**2)),
                          scale = torch.sqrt(torch.log(1 + sigma**2/mu**2)))
