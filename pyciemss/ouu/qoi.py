from typing import Dict, List, Optional

import numpy as np
import torch


def obs_nday_average_qoi(
    samples: Dict[str, torch.Tensor],
    contexts: List,
    *,
    ndays: int = 7,
    qoi_start_time: Optional[int] = None,
    qoi_end_time: Optional[int] = None,
) -> np.ndarray:
    """
    Return estimate of last n-day average of each sample.
    samples is is the output from a Pyro Predictive object.
    samples[VARIABLE] is expected to have dimension (nreplicates, ntimepoints)
    Note: last ndays timepoints is assumed to represent last n-days of simulation.
    ndays = 1 leads to using the value at the end of the simulation.
    """
    dataQoI = samples[contexts[0]][..., 0, :].detach().numpy()
    if qoi_start_time is not None:
        id_start = torch.where(samples["logging_times"] >= qoi_start_time)[0][0].item()
        dataQoI = dataQoI[:, id_start-1:]
    if qoi_end_time is not None:
        id_end = torch.where(samples["logging_times"] <= qoi_end_time)[0][0].item()
        dataQoI = dataQoI[:, :id_end]
    return np.mean(dataQoI[:, -ndays:], axis=1)


def obs_max_qoi(
    samples: Dict[str, torch.Tensor],
    contexts: List,
    *,
    qoi_start_time: Optional[int] = None,
    qoi_end_time: Optional[int] = None,
) -> np.ndarray:
    """
    Return maximum value over simulated time.
    samples is is the output from a Pyro Predictive object.
    samples[VARIABLE] is expected to have dimension (nreplicates, ntimepoints)
    """
    dataQoI = samples[contexts[0]][..., 0, :].detach().numpy()
    if qoi_start_time is not None:
        id_start = torch.where(samples["logging_times"] >= qoi_start_time)[0][0].item()
        dataQoI = dataQoI[:, id_start-1:]
    if qoi_end_time is not None:
        id_end = torch.where(samples["logging_times"] <= qoi_end_time)[0][0].item()
        dataQoI = dataQoI[:, :id_end]
    return np.max(dataQoI, axis=1)
