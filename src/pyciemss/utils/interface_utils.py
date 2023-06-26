import pandas as pd
import numpy as np

import torch

import csv
from typing import Dict


def convert_to_output_format(samples: Dict[str, torch.Tensor]) -> pd.DataFrame:
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
    d = {
        **d,
        **{
            k: np.repeat(v, num_timepoints)
            for k, v in pyciemss_results["parameters"].items()
        },
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
