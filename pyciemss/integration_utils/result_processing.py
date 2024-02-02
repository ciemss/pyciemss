from typing import Any, Dict, Optional, Iterable

import numpy as np
import pandas as pd
import torch


def prepare_interchange_dictionary(
    samples: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    processed_samples = convert_to_output_format(samples)

    result = {"data": processed_samples, "unprocessed_result": samples}

    return result


def convert_to_output_format(
    samples: Dict[str, torch.Tensor],
    time_unit: Optional[str] = None,
    timepoints: Optional[Iterable[float]] = None,
) -> pd.DataFrame:
    """
    Convert the samples from the Pyro model to a DataFrame in the TA4 requested format.
    """

    pyciemss_results: Dict[str, Dict[str, torch.Tensor]] = {
        "parameters": {},
        "states": {},
    }

    for name, sample in samples.items():
        if sample.ndim == 1:
            # Any 1D array is a sample from the distribution over parameters.
            # Any 2D array is a sample from the distribution over states, unless it's a model weight.
            name = name + "_param" if not name.endswith("_param") else name
            pyciemss_results["parameters"][name] = (
                sample.data.detach().cpu().numpy().astype(np.float64)
            )
        else:
            name = (
                name + "_state"
                if not (
                    name.endswith("_state")
                )
                else name
            )
            pyciemss_results["states"][name] = (
                sample.data.detach().cpu().numpy().astype(np.float64)
            )

    num_samples, num_timepoints = next(
        iter(pyciemss_results["states"].values())
    ).shape
    output = {
        "timepoint_id": np.tile(np.array(range(num_timepoints)), num_samples),
        "sample_id": np.repeat(np.array(range(num_samples)), num_timepoints),
    }

    # Parameters
    output = {
        **output,
        **{
            k: np.repeat(v, num_timepoints)
            for k, v in pyciemss_results["parameters"].items()
        },
    }

    # Solution (state variables)
    output = {
        **output,
        **{
            k: np.squeeze(v.reshape((num_timepoints * num_samples, 1)))
            for k, v in pyciemss_results["states"].items()
        },
    }

    result = pd.DataFrame(output)
    if time_unit is not None:
        all_timepoints = result["timepoint_id"].map(lambda v: timepoints[v])
        result = result.assign(**{f"timepoint_{time_unit}": all_timepoints})

    return result
