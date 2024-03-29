import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

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
    samples = {k: (v.squeeze() if len(v.shape) > 2 else v) for k, v in samples.items()}

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

    # Remove intervention data from the samples...
    intervention_data = {
        k: v for k, v in samples.items() if "parameter_intervention_value" in k
    }
    intervention_times = {
        k: v for k, v in samples.items() if "parameter_intervention_time" in k
    }
    samples = {
        k: v
        for k, v in samples.items()
        if k not in intervention_data.keys() and k not in intervention_times.keys()
    }

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

    # Set values to reflect interventions in time-order
    for name, values in sorted(
        intervention_data.items(), key=lambda item: int(item[0].split("_")[-1])
    ):
        result = set_intervention_values(result, name, values, intervention_times)

    return result


# --- Intervention weaving utilities ----
def get_times_for(intervention: str, intervention_times: Mapping[str, Iterable[float]]):
    """intervention -- Full name of the intervention entry (will be parsed to get the time)
    intervention_times --
    """
    time_id = f"_{intervention.split('_')[-1]}"
    valid = [o for o in intervention_times.keys() if o.endswith(time_id)]

    if len(valid) > 1:
        raise ValueError(f"Time-alignment was not clear for {intervention}")
    if len(valid) == 0:
        raise KeyError(f"Time-alignment not found for {intervention}")

    return intervention_times[valid[0]]


def find_target_col(var: str, options: List[str]):
    """
    Find the column that corresponds to the var
    var -- The parsed variable name
    options -- Column names to search for the variable name
    """
    # TODO: This "underscore-trailing-name matching" seems very fragile....
    #       It is done this way since you can intervene on params & states
    #       and that will match either.
    pattern = re.compile(f"(?:^|_){var}_(state|param)")
    options = [c for c in options if pattern.search(c)]
    if len(options) == 0:
        raise KeyError(f"No target column match found for '{var}'.")
    if len(options) > 1:
        raise ValueError(
            f"Could not uniquely determine target column for '{var}'.  Found: {options}"
        )
    return options[0]


def set_intervention_values(
    df: pd.DataFrame,
    intervention: str,
    intervention_values: torch.Tensor,
    intervention_times: Dict[str, torch.Tensor],
):
    """
    df -- Results similar to those returned from 'sample'
          (but not including columns that describe parameter interventions)
    intervention -- Target intervention name
    intervention_values -- The values for the target intervention
    intervention_times -- The dictionary of all intervention
    """
    times = get_times_for(intervention, intervention_times)
    target_var = "_".join(intervention.split("_")[3:-1])
    target_col = find_target_col(target_var, df.columns)
    time_col = [
        c for c in df.columns if c.startswith("timepoint_") and c != "timepoint_id"
    ][0]

    def rework(group):
        mask = group[time_col] < float(times[group.name])
        replacement_value = float(intervention_values[group.name])
        group[target_col] = group[target_col].where(mask, replacement_value)

        return group

    return df.groupby("sample_id").apply(rework).reset_index(drop=True)
