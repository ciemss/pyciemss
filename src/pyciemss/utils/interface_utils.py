import pandas as pd
import numpy as np

import torch

import csv
from typing import Dict, Optional, Iterable


def convert_to_output_format(
    samples: Dict[str, torch.Tensor],
    timepoints: Iterable[float],
    interventions: Optional[Dict[str, torch.Tensor]] = None,
) -> pd.DataFrame:
    """
    Convert the samples from the Pyro model to a DataFrame in the TA4 requested format.
    """

    pyciemss_results = {"parameters": {}, "states": {}}

    for name, sample in samples.items():
        if sample.ndim == 1:
            # Any 1D array is a sample from the distribution over parameters.
            # Any 2D array is a sample from the distribution over states.
            name = name + "_param"
            pyciemss_results["parameters"][name] = (
                sample.data.detach().cpu().numpy().astype(np.float64)
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
            **assign_parameters_to_timepoints(interventions, timepoints, pyciemss_results["parameters"])
        }

    # Solution (state variables)
    d = {
        **d,
        **{
            k: np.squeeze(v.reshape((num_timepoints * num_samples, 1)))
            for k, v in pyciemss_results["states"].items()
        },
    }

    return pd.DataFrame(d)


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

def intervention_to_interval(interventions:dict, initial_params: dict) -> dict:
    """Convert intervention dict and initial_params dict to dict of intervals.

    :param interventions: dict of interventions keyed by parameter name with values (start, value)
    :param initial_values: dict of initial values keyed by parameter name with the unintervened parameter value
    :return: dict keyed by param where the values lists of intervals and values sorted by start time.
    """
    param_dict = {param: [dict(start=0, end=np.inf, expected_value=value)]
                  for param, value in initial_params.items()}
    for i, (start, param, value) in enumerate(sorted(interventions)):
        param_dict[param][-1]['end'] = start
        param_dict[param].append(dict(start=start, end=np.inf, expected_value=[value]*len(initial_values[param])))
    return {
        k: sorted(v, key=lambda x: x['start'])
        for k, v in param_dict.items()
    }     

def assign_parameters_to_timepoints(interventions: dict, timepoints: Iterable[float], initial_values: dict) -> dict:
    """Generate a len(timepoints)*len(samples) array of parameter values for each parameter in interventions.

    :param interventions: dict of interventions keyed by parameter name with values (start, value)
    :param timepoints: iterable of timepoints
    :param initial_values: dict of initial values keyed by parameter name with the unintervened parameter value
    :return: dict keyed by param where the values are lists of (timepoint, value) pairs sorted by timepoint
    """
    # transform samples into intervals and parameter values
    param_interval_dict = intervention_to_interval(interventions, initial_values)
    result = {}
    for param, interval_dict in param_interval_dict.items():
        intervals = [(d['start'], d['end']) for d in interval_dict]
        values_sets = [d['expected_value'] for  d in interval_dict]

        # generate list of parameter values at each timepoint
        result[param] = []
        for values in zip(*values_sets):
            for t in timepoints:
                i = bisect.bisect(intervals, (t,)) - 1
                if 0 <= i < len(intervals) and intervals[i][0] <= t < intervals[i][1]:
                    result[param].append(values[i])
                else:
                    result[param].append(None)
    return result

