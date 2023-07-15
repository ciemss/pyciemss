import pandas as pd
import numpy as np
import bisect
import torch
import csv
from typing import Dict, Optional, Iterable, Callable


def convert_to_output_format(
    samples: Dict[str, torch.Tensor],
    timepoints: Iterable[float],
    interventions: Optional[Dict[str, torch.Tensor]] = None,
    *,
    time_unit: Optional[str] = "(unknown)",
    ensemble_quantiles: Optional[bool] = False,
    alpha_qs: Optional[Iterable[float]] = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99],
    num_ensemble_quantiles: Optional[int] = 0,
    stacking_order: Optional[str] = "timepoints",
    observables: Optional[Dict[str, Callable]] = None,
) -> pd.DataFrame:
    """
    Convert the samples from the Pyro model to a DataFrame in the TA4 requested format.

    time_unit -- Label timepoints in a semantically relevant way `timepoint_<time_unit>`.
       If None, a `timepoint_<time_unit>` field is not provided.
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

    if observables is not None:
        expression_vars = {
            k[:-4]: torch.squeeze( torch.tensor(d[k]), dim=-1)
            for k in pyciemss_results["states"].keys()
        }
        print(f"expression_vars: {expression_vars}\n states: {pyciemss_results['states'].keys()}")
        d = {
            **d,
            **{
                f"{observable_id}_obs": torch.squeeze( expression(**expression_vars))
                for observable_id, expression in observables.items()
            },
        }
    result = pd.DataFrame(d)
    if time_unit is not None:
        all_timepoints = result["timepoint_id"].map(lambda v: timepoints[v])
        result = result.assign(**{f"timepoint_{time_unit}": all_timepoints})

    if ensemble_quantiles:
        key_list = ["timepoint_id", "target", "type", "quantile", "value"]
        q = {k: [] for k in key_list}
        if alpha_qs is None:
            alpha_qs = np.linspace(0, 1, num_ensemble_quantiles)
            alpha_qs[0] = 0.01
            alpha_qs[-1] = 0.99
        else:
            num_ensemble_quantiles = len(alpha_qs)
        
        # Solution (state variables)
        for k, v in pyciemss_results["states"].items():
            q_vals = np.quantile(v, alpha_qs, axis=0)
            k = k.replace("_sol","")
            if stacking_order == "timepoints":
                # Keeping timepoints together
                q["timepoint_id"].extend(list(np.repeat(np.array(range(num_timepoints)), num_ensemble_quantiles)))
                q["target"].extend([k]*num_timepoints*num_ensemble_quantiles)
                q["type"].extend(["quantile"]*num_timepoints*num_ensemble_quantiles)
                q["quantile"].extend(list(np.tile(alpha_qs, num_timepoints)))
                q["value"].extend(list(np.squeeze(q_vals.T.reshape((num_timepoints * num_ensemble_quantiles, 1)))))
            elif stacking_order == "quantiles":
                # Keeping quantiles together
                q["timepoint_id"].extend(list(np.tile(np.array(range(num_timepoints)), num_ensemble_quantiles)))
                q["target"].extend([k]*num_timepoints*num_ensemble_quantiles)
                q["type"].extend(["quantile"]*num_timepoints*num_ensemble_quantiles)
                q["quantile"].extend(list(np.repeat(alpha_qs, num_timepoints)))
                q["value"].extend(list(np.squeeze(q_vals.reshape((num_timepoints * num_ensemble_quantiles, 1)))))
            else:
                raise Exception("Incorrect input for stacking_order.")
        result_q = pd.DataFrame(q)
        if time_unit is not None:
            all_timepoints = result_q["timepoint_id"].map(lambda v: timepoints[v])
            result_q = result_q.assign(**{f"timepoint_{time_unit}": all_timepoints})            
        return result, result_q
    else:
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
            # use float for the timestep, and convert the values in the dictionary to float only if not NaN or NA
            result.append((float(row[0]), {k: float(v) for k, v in data_dict.items() if not(v=='' or v=='NaN')}))
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


def solutions_to_observations(timepoints: Iterable, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Convert pyciemss outputs to data observations."""
    # Use groupby to create separate DataFrames
    grouped = df.groupby(level=1)

    # Create a dictionary of dataframes
    outputs = {level: group for level, group in grouped}
    observations = dict()
    for idx, observation in outputs.items():
        observation = observation.drop([k for k in observation.columns if '_sol' != k[-4:]],
                                      axis=1)
        observation = observation.rename(columns={k: k[:-4] for k in observation.columns})
        observation['Timestep'] = timepoints

        observations[idx] = observation[['Timestep'] + [c for c in observation.columns[:-1]]]
    return observations
      
