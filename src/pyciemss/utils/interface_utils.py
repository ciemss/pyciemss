import pandas as pd
import numpy as np
import bisect
import torch

import csv
from typing import Dict, Optional, Iterable


def convert_to_output_format(
    samples: Dict[str, torch.Tensor],
    timepoints: Iterable[float],
    interventions: Optional[Dict[str, torch.Tensor]] = None,
    *,
    time_unit: Optional[str] = "(unknown)",
) -> pd.DataFrame:
    """
    Convert the samples from the Pyro model to a DataFrame in the TA4 requested format.

    time_unit -- Label timepoints in a semantically relevant way `timepoint_<time_unit>`.
       If ommited, a `timepoint_<time_unit>` field is not provided.
    """

    pyciemss_results = {"parameters": {}, "states": {}}

    for name, sample in samples.items():
        if sample.ndim == 1:
            # Any 1D array is a sample from the distribution over parameters.
            # Any 2D array is a sample from the distribution over states, unless it's a model weight.
            name = name + "_param"
            pyciemss_results["parameters"][name] = (
                sample.data.detach().cpu().numpy().astype(np.float64)
            )
        elif name == "model_weights":
            n_models = sample.shape[1]
            for i in range(n_models):
                pyciemss_results["parameters"][f"model_{i}_weight"] = (
                    sample[:, i].data.detach().cpu().numpy().astype(np.float64)
                )
        else:
            pyciemss_results["states"][name] = (
                sample.data.detach().cpu().numpy().astype(np.float64)
            )

    num_samples, num_timepoints = next(iter(pyciemss_results["states"].values())).shape
    d = {
        "timepoint_id": np.tile(np.array(range(num_timepoints)), num_samples),
        "sample_id": np.repeat(np.array(range(num_samples)), num_timepoints),
    }

    # Parameters
    if interventions is None:
        d = {
            **d,
            **{
                k: np.repeat(v, num_timepoints)
                for k, v in pyciemss_results["parameters"].items()
            },
        }
    else:
        d = {
            **d,
            **assign_interventions_to_timepoints(
                interventions, timepoints, pyciemss_results["parameters"]
            ),
        }

    # Solution (state variables)
    d = {
        **d,
        **{
            k: np.squeeze(v.reshape((num_timepoints * num_samples, 1)))
            for k, v in pyciemss_results["states"].items()
        },
    }

    result = pd.DataFrame(d)
    if time_unit is not None:
        all_timepoints = result["timepoint_id"].map(lambda v: timepoints[v])
        result = result.assign(**{f"timepoint_{time_unit}": all_timepoints})

    return result


def csv_to_list(filename):
    result = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            # zip function pairs header elements and row elements together
            # it will ignore extra values in either the header or the row
            data_dict = dict(zip(header[1:], row[1:]))
            # use float for the timestep, and convert the values in the dictionary to float
            result.append((float(row[0]), {k: float(v) for k, v in data_dict.items()}))
    return result


def interventions_and_sampled_params_to_interval(
    interventions: dict, sampled_params: dict
) -> dict:
    """Convert interventions and sampled parameters to dict of intervals.

    :param interventions: dict keyed by parameter name where each value is a tuple (intervention_time, value)
    :param sampled_params: dict keyed by param where each value is an array of sampled parameter values
    :return: dict keyed by param where the values lists of intervals and values sorted by start time
    """
    # assign each sampled parameter to an infinite interval
    param_dict = {
        param: [dict(start=-np.inf, end=np.inf, param_values=value)]
        for param, value in sampled_params.items()
    }

    # sort the interventions by start time
    for start, param, intervention_value in sorted(interventions):
        # update the end time of the previous interval
        param_dict[f"{param}_param"][-1]["end"] = start

        # add new interval and broadcast the intevention value to the size of the sampled parameters
        param_dict[f"{param}_param"].append(
            dict(
                start=start,
                end=np.inf,
                param_values=[intervention_value]
                * len(sampled_params[f"{param}_param"]),
            )
        )

    # sort intervals by start time
    return {k: sorted(v, key=lambda x: x["start"]) for k, v in param_dict.items()}


def assign_interventions_to_timepoints(
    interventions: dict, timepoints: Iterable[float], sampled_params: dict
) -> dict:
    """Assign the value of each parameter to every timepoint, taking into account interventions.

    :param interventions: dict keyed by parameter name where each value is a tuple (intervention_time, value)
    :param timepoints: iterable of timepoints
    :param sampled_params: dict keyed by parameter name where each value is an array of sampled parameter values
    :return: dict keyed by param where the values are sorted by sample then timepoint
    """
    # transform interventions and sampled parameters into intervals
    param_interval_dict = interventions_and_sampled_params_to_interval(
        interventions, sampled_params
    )
    result = {}
    for param, interval_dict in param_interval_dict.items():
        intervals = [(d["start"], d["end"]) for d in interval_dict]
        param_values = [d["param_values"] for d in interval_dict]

        # generate list of parameter values at each timepoint
        result[param] = []
        for values in zip(*param_values):
            for t in timepoints:
                # find the interval that contains the timepoint
                i = bisect.bisect(intervals, (t,)) - 1
                result[param].append(values[i])
    return result
