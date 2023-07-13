import os
import urllib.request, json 
import pandas as pd
from IPython.display import HTML
from IPython import display
from pyciemss.PetriNetODE.interfaces import (
    load_and_sample_petri_model,
    load_and_calibrate_and_sample_petri_model,
    load_petri_model,
    setup_petri_model,
    sample
)
import numpy as np
from typing import Iterable
from pyciemss.utils.interface_utils import (
    assign_interventions_to_timepoints,
    interventions_and_sampled_params_to_interval,
    convert_to_output_format
)
from pyciemss.utils import get_tspan
import matplotlib.pyplot as plt
import torch
from torch import tensor

from toronto_ensemble_challenge_utils import get_case_hosp_death_data

infectious_period = 7
covid_data_df = get_case_hosp_death_data(US_region = "MI", infectious_period = infectious_period, make_csv=False)

# S E I R D
mapped_data = {}
mapped_data["timepoints"] = list(range(len(covid_data_df.index)))
mapped_data["I"] = covid_data_df["case_census"]
mapped_data["D"] = covid_data_df["cumulative_deaths"]
# write to CSV file
mapped_data_df = pd.DataFrame(mapped_data)
mapped_data_df.to_csv("mapped_data.csv", index=False)

num_samples = 1
# timepoints already defined
SEIRD_model_path = "scenario1_a.json" # model Q1a

# Run the calibration and sampling
calibrated_samples = load_and_calibrate_and_sample_petri_model(petri_model_or_path=SEIRD_model_path,data_path="mapped_data.csv",num_samples=num_samples,\
    timepoints=mapped_data["timepoints"],verbose=True)