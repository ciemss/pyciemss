from pyciemss.visuals import checks, plots
from pathlib import Path
import unittest
import numpy as np
import xarray as xr
import os
import pandas as pd

from pyciemss.utils import plot_utils
import torch

from pyciemss.visuals import plots, vega, trajectories
from pyciemss.utils import get_tspan
from pyciemss.utils.interface_utils import convert_to_output_format
import json

create_png_plots = True

def tensor_load(path):
    """Load tensor data """
    with open(path) as f:
        data = json.load(f)

    data = {k: torch.from_numpy(np.array(v)) for k, v in data.items()}

    return data


def by_key_value(targets, key, value):
    """Return entry by key value from target if equals value"""
    for entry in targets:
        if entry[key] == value:
            return entry



_data_root = Path(__file__).parent.parent / "data"

save_png = (
    Path(__file__).parent.parent /  "test_utils" / "reference_images" 
)

class TestPlotUtils(unittest.TestCase):
    def setUp(self):
        """ Get starting values for trajectory plot with rabbits and wolves"""
        self.tspan = get_tspan(1, 50, 500).detach().numpy()
        self.nice_labels = {"Rabbits_sol": "Rabbits", "Wolves_sol": "Wolves"}

        self.dists = convert_to_output_format(
            tensor_load(_data_root / "prior_samples.json"),
            self.tspan,
            time_unit="notional",
        )

        exemplars = self.dists[self.dists["sample_id"] == 0]

        wolves = exemplars.set_index("timepoint_notional")["Wolves_sol"].rename(
            "Wolves Example"
        )
        rabbits = exemplars.set_index("timepoint_notional")["Rabbits_sol"].rename(
            "Rabbits Example"
        )
        self.traces = pd.concat([wolves, rabbits], axis="columns")

        self.observed_trajectory = convert_to_output_format(
            tensor_load(_data_root / "observed_trajectory.json"),
            self.tspan,
            time_unit="years",
        )

        self.observed_points = (
            self.observed_trajectory.rename(columns={"Rabbits_sol": "Rabbits Samples"})
            .drop(
                columns=[
                    "Wolves_sol",
                    "alpha_param",
                    "beta_param",
                    "delta_param",
                    "gamma_param",
                ]
            )
            .iloc[::10]
        )

    def test_plot_predictive(self):
        import pdb; pdb.set_trace()
        ##TODO requires tensor input, should that be the case?
        ax = plot_utils.plot_predictive(tensor_load(_data_root / "prior_samples.json"), torch.tensor(self.tspan), vars=["Rabbits_sol"])
        ax = plot_utils.plot_trajectory(tensor_load(_data_root / "observed_trajectory.json"), torch.tensor(self.tspan), ax=ax, vars=["Rabbits_sol"])

        if create_png_plots:
            with open(os.path.join(save_png,  "plot_predictive.png"), "wb") as f:
                ax.figure.savefig(f)


