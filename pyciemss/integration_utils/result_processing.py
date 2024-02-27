from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd
import torch

from pyciemss.visuals import plots


def prepare_interchange_dictionary(
    samples: Dict[str, torch.Tensor],
    time_unit: Optional[str] = None,
    timepoints: Optional[Iterable[float]] = None,
    visual_options: Union[None, bool, Dict[str, Any]] = None,
) -> Dict[str, Any]:
    processed_samples = convert_to_output_format(
        samples, time_unit=time_unit, timepoints=timepoints
    )

    result = {"data": processed_samples, "unprocessed_result": samples}

    if visual_options:
        visual_options = {} if visual_options is True else visual_options
        schema = plots.trajectories(processed_samples, **visual_options)
        result["schema"] = schema

    return result


def convert_to_output_format(
    samples: Dict[str, torch.Tensor],
    *,
    time_unit: Optional[str] = None,
    timepoints: Optional[Iterable[float]] = None,
) -> pd.DataFrame:
    """
    Convert the samples from the Pyro model to a DataFrame in the TA4 requested format.
    """

    if time_unit is not None and timepoints is None:
        raise ValueError("`timepoints` must be supplied when a `time_unit` is supplied")

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
        elif sample.ndim == 2 and name == "model_weights":
            for i in range(sample.shape[1]):
                pyciemss_results["parameters"][f"model_{i}/weight_param"] = (
                    sample[:, i].data.detach().cpu().numpy().astype(np.float64)
                )
        else:
            name = name + "_state" if not (name.endswith("_state")) else name
            pyciemss_results["states"][name] = (
                sample.data.detach().cpu().numpy().astype(np.float64)
            )

    num_samples, num_timepoints = next(iter(pyciemss_results["states"].values())).shape
    output = {
        "timepoint_id": np.tile(np.array(range(num_timepoints)), num_samples),
        "sample_id": np.repeat(np.array(range(num_samples)), num_timepoints),
    }

    if timepoints is not None:
        timepoints = [*timepoints]
        label = "timepoint_unknown" if time_unit is None else f"timepoint_{time_unit}"
        output[label] = np.array(float(timepoints[v]) for v in output["timepoint_id"])

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

    return result
