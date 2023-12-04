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


_data_root = Path(__file__).parent.parent / "data"

save_png = (
    Path(__file__).parent.parent /  "test_utils" / "reference_images" 
)

class TestPlotUtils(unittest.TestCase):
    def setUp(self):
        """ Get starting values for trajectory plot with rabbits and wolves"""
        self.tspan = get_tspan(1, 50, 500).detach().numpy()


    def test_plot_all(self):
        ##TODO requires tensor input, should that be the case?
        ax = plot_utils.plot_predictive(tensor_load(_data_root / "prior_samples.json"), torch.tensor(self.tspan), vars=["Rabbits_sol"])
        ax = plot_utils.plot_trajectory(tensor_load(_data_root / "observed_trajectory.json"), torch.tensor(self.tspan), ax=ax, vars=["Rabbits_sol"])
        ax = plot_utils.plot_intervention_line(3, ax=ax)
        if create_png_plots:
            with open(os.path.join(save_png,  "plot_all.png"), "wb") as f:
                ax.figure.savefig(f)

    def test_plot_predictive(self):
        ax = plot_utils.plot_predictive(tensor_load(_data_root / "prior_samples.json"), torch.tensor(self.tspan),  tmin=10, alpha=1, color="green", ptiles=[0.10,0.90], vars=["Rabbits_sol"])
        if create_png_plots:
            with open(os.path.join(save_png,  "plot_predictive.png"), "wb") as f:
                ax.figure.savefig(f)

    def test_plot_trajectory(self):
        ax = plot_utils.plot_trajectory(tensor_load(_data_root / "observed_trajectory.json"), torch.tensor(self.tspan),  color='red', alpha=2, lw=2, marker='-', label="My label",  vars=["Rabbits_sol"])
        if create_png_plots:
            with open(os.path.join(save_png,  "plot_trajectory.png"), "wb") as f:
                ax.figure.savefig(f)

    def test_plot_intervention_line(self):
        ax = plot_utils.plot_intervention_line(10)
        if create_png_plots:
            with open(os.path.join(save_png,  "plot_intervention_line.png"), "wb") as f:
                ax.figure.savefig(f)

    def test_plot_ouu_risk(self):
        [ax, cax] = plot_utils.plot_ouu_risk(tensor_load(_data_root / "prior_samples.json"))
        if create_png_plots:
            with open(os.path.join(save_png,  "plot_ouu_risk_ax.png"), "wb") as f:
                ax.figure.savefig(f)
            with open(os.path.join(save_png,  "plot_ouu_risk_cax.png"), "wb") as f:
                cax.figure.savefig(f)
            

