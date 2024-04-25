import warnings
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from pyciemss.visuals import plots

DEFAULT_ALPHA_QS = [
    0.01,
    0.025,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.975,
    0.99,
]


def prepare_interchange_dictionary(
    samples: Dict[str, torch.Tensor],
    time_unit: Optional[str] = None,
    timepoints: Optional[torch.Tensor] = None,
    visual_options: Union[None, bool, Dict[str, Any]] = None,
    ensemble_quantiles: bool = False,
    alpha_qs: Optional[List[float]] = DEFAULT_ALPHA_QS,
    stacking_order: str = "timepoints",
) -> Dict[str, Any]:
    samples = {k: (v.squeeze() if len(v.shape) > 2 else v) for k, v in samples.items()}

    processed_samples, quantile_results = convert_to_output_format(
        samples,
        time_unit=time_unit,
        timepoints=timepoints,
        ensemble_quantiles=ensemble_quantiles,
        alpha_qs=alpha_qs,
        stacking_order=stacking_order,
    )

    result = {"data": processed_samples, "unprocessed_result": samples}
    if ensemble_quantiles:
        result["ensemble_quantiles"] = quantile_results

    if visual_options:
        visual_options = {} if visual_options is True else visual_options
        schema = plots.trajectories(processed_samples, **visual_options)
        result["schema"] = schema

    return result


def convert_to_output_format(
    samples: Dict[str, torch.Tensor],
    *,
    time_unit: Optional[str] = None,
    timepoints: Optional[torch.Tensor] = None,
    ensemble_quantiles: bool = False,
    alpha_qs: Optional[List[float]] = None,
    stacking_order: str = "timepoints",
) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None]]:
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

    pyciemss_results: Dict[str, Dict[str, np.ndarray]] = {
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
        label = "timepoint_unknown" if time_unit is None else f"timepoint_{time_unit}"
        output[label] = np.array(
            float(timepoints[v].item()) for v in output["timepoint_id"]
        )

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

    if ensemble_quantiles:
        result_quantiles = make_quantiles(
            pyciemss_results,
            alpha_qs=alpha_qs,
            time_unit=time_unit,
            timepoints=timepoints,
            stacking_order=stacking_order,
        )
    else:
        result_quantiles = None

    return result, result_quantiles


def make_quantiles(
    pyciemss_results: Dict[str, Dict[str, np.ndarray]],
    *,
    alpha_qs: Optional[List[float]] = None,
    time_unit: Optional[str] = None,
    timepoints: Optional[torch.Tensor] = None,
    stacking_order: str = "timepoints",
) -> Union[pd.DataFrame, None]:
    """Make quantiles for each timepoint"""
    _, num_timepoints = next(iter(pyciemss_results["states"].values())).shape
    key_list = ["timepoint_id", "output", "type", "quantile", "value"]
    q: Dict[str, List] = {k: [] for k in key_list}
    if alpha_qs is not None:
        num_quantiles = len(alpha_qs)

        # Solution (state variables)
        for k, v in pyciemss_results["states"].items():
            q_vals = np.quantile(v, alpha_qs, axis=0)
            k = k.replace("_sol", "")
            if stacking_order == "timepoints":
                # Keeping timepoints together
                q["timepoint_id"].extend(
                    list(np.repeat(np.array(range(num_timepoints)), num_quantiles))
                )
                q["output"].extend([k] * num_timepoints * num_quantiles)
                q["type"].extend(["quantile"] * num_timepoints * num_quantiles)
                q["quantile"].extend(list(np.tile(alpha_qs, num_timepoints)))
                q["value"].extend(
                    list(
                        np.squeeze(
                            q_vals.T.reshape((num_timepoints * num_quantiles, 1))
                        )
                    )
                )
            elif stacking_order == "quantiles":
                # Keeping quantiles together
                q["timepoint_id"].extend(
                    list(np.tile(np.array(range(num_timepoints)), num_quantiles))
                )
                q["output"].extend([k] * num_timepoints * num_quantiles)
                q["type"].extend(["quantile"] * num_timepoints * num_quantiles)
                q["quantile"].extend(list(np.repeat(alpha_qs, num_timepoints)))
                q["value"].extend(
                    list(
                        np.squeeze(q_vals.reshape((num_timepoints * num_quantiles, 1)))
                    )
                )
            else:
                raise Exception("Incorrect input for stacking_order.")

        result_quantiles = pd.DataFrame(q)
        if timepoints is not None:
            all_timepoints = result_quantiles["timepoint_id"].map(
                lambda v: timepoints[v].item()
            )
            result_quantiles = result_quantiles.assign(
                **{f"number_{time_unit}": all_timepoints}
            )
            result_quantiles = result_quantiles[
                [
                    "timepoint_id",
                    f"number_{time_unit}",
                    "output",
                    "type",
                    "quantile",
                    "value",
                ]
            ]
        else:
            result_quantiles = None
    return result_quantiles


def cdc_format(
    q_ensemble_input: pd.DataFrame,
    solution_string_mapping: Dict[str, str],
    *,
    time_unit: Optional[str] = None,
    forecast_start_date: Optional[str] = None,
    location: Optional[str] = None,
    drop_column_names: List[str] = [
        "timepoint_id",
        "output",
    ],
    train_end_point: Optional[float] = None,
) -> pd.DataFrame:
    """
    Reformat the quantiles pandas dataframe file to CDC ensemble forecast format
    Note that solution_string_mapping maps name of states/observables in the dictionary key to the dictionary value
    and also drops any states/observables not available in the dictionary keys.
    forecast_start_date is the date of last observed data.
    """
    q_ensemble_data = deepcopy(q_ensemble_input)
    if time_unit != "days" or time_unit is None:
        warnings.warn(
            "cdc_format only works for time_unit=days"
            "time_unit will default to days and overwrite previous time_unit."
        )
        q_ensemble_data.rename(columns={"number_None": "number_days"}, inplace=True)
        if "number_days" not in q_ensemble_data:
            raise ValueError("time_unit can only support days")
        time_unit = "days"

    if train_end_point is None:
        q_ensemble_data["Forecast_Backcast"] = "Forecast"
        number_data_days = 0.0
    else:
        q_ensemble_data["Forecast_Backcast"] = np.where(
            q_ensemble_data[f"number_{time_unit}"] > train_end_point,
            "Forecast",
            "Backcast",
        )
        # Number of days for which data is available
        number_data_days = max(
            q_ensemble_data[
                q_ensemble_data["Forecast_Backcast"].str.contains("Backcast")
            ][f"number_{time_unit}"]
        )
    drop_column_names.extend(["Forecast_Backcast"])
    # Subtracting number of backast days from number_days
    q_ensemble_data[f"number_{time_unit}"] = (
        q_ensemble_data[f"number_{time_unit}"] - number_data_days
    )
    # Drop rows that are backcasting
    q_ensemble_data = q_ensemble_data[
        ~q_ensemble_data["Forecast_Backcast"].str.contains("Backcast")
    ]
    # Changing name of state according to user provided strings
    if solution_string_mapping:
        # Drop rows that are not present in the solution_string_mapping keys
        q_ensemble_data = q_ensemble_data[
            q_ensemble_data["output"].str.contains(
                "|".join(solution_string_mapping.keys())
            )
        ]
        for k, v in solution_string_mapping.items():
            q_ensemble_data["output"] = q_ensemble_data["output"].replace(k, v)

    # Creating target column
    q_ensemble_data["target"] = (
        q_ensemble_data[f"number_{time_unit}"].astype("string")
        + " days ahead "
        # + q_ensemble_data["inc_cum"]
        + " "
        + q_ensemble_data["output"]
    )

    # Add dates
    if forecast_start_date:
        q_ensemble_data["forecast_date"] = pd.to_datetime(
            forecast_start_date, format="%Y-%m-%d", errors="ignore"
        )
        q_ensemble_data["target_end_date"] = q_ensemble_data["forecast_date"].combine(
            q_ensemble_data[f"number_{time_unit}"],
            lambda x, y: x + pd.DateOffset(days=int(y)),
        )
    # Add location column
    if location:
        q_ensemble_data["location"] = location
    # Dropping columns specified by user
    if drop_column_names:
        q_ensemble_data = q_ensemble_data.drop(columns=drop_column_names)
    return q_ensemble_data


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
    target_col = f"persistent_{target_var}_param"

    if target_col not in df.columns:
        raise KeyError(f"Could not find target column for '{target_var}'")

    time_col = [
        c for c in df.columns if c.startswith("timepoint_") and c != "timepoint_id"
    ][0]

    def rework(group):
        mask = group[time_col] < float(times[group.name])
        replacement_value = float(intervention_values[group.name])
        group[target_col] = group[target_col].where(mask, replacement_value)

        return group

    return df.groupby("sample_id").apply(rework).reset_index(drop=True)
