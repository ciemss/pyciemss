import os

import torch
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime

from pyciemss.PetriNetODE.interfaces import load_petri_model
from pyciemss.PetriNetODE.interfaces import sample as sample_petri
from pyciemss.PetriNetODE.interfaces import setup_model as setup_model_petri
from pyciemss.PetriNetODE.interfaces import intervene as intervene_petri
from pyciemss.PetriNetODE.interfaces import calibrate as calibrate_petri

from pyciemss.Ensemble.interfaces import setup_model as setup_model_ensemble
from pyciemss.Ensemble.interfaces import reset_model as reset_model_ensemble
from pyciemss.Ensemble.interfaces import intervene as intervene_ensemble
from pyciemss.Ensemble.interfaces import sample as sample_ensemble
from pyciemss.Ensemble.interfaces import calibrate as calibrate_ensemble
from pyciemss.Ensemble.interfaces import optimize as optimize_ensemble

def create_synth_data(weights, start_date, modelA_sample, modelB_sample=None, modelC_sample=None):
    # Function that takes in any number (up to 3) of the previously defined model samples along with a set of weights,
    # and returns the weighted sum of the sample output, aka the synthetic data DataFrame: synth_data_df, as well as the
    # dictionary sample_data containing the original model output.
    # Inputs:
    #       weights = weights!

    # Note that hard-coded into the function are the start date 11/29/2021, and 100 as the number of time points

    # must include one weight per model, weights must be positive and sum to 1

    num_models = len(weights)
    sample_array = [modelA_sample, modelB_sample, modelC_sample]

    # Create synth_data dictionary, a dictionary of dictionaries which has models as keys, and dictionaries
    # containing their outputs (mapped to keys: Cases, Hospitalizations, and Deaths) as values
    sample_data = {}
    for i in range(num_models):

        # Map state variables to data and save output to synth_data dictionary
        state_var_sol_dict = {}
        for n in sample_array[i].keys():
            if "population" in n:
                state_var_sol_dict[n[0:-4]] = sample_array[i][n]

        if "symptomatic_population" in state_var_sol_dict.keys():
            sample_data['model' + str(i)] = solution_mapping_symp(state_var_sol_dict)

        else:
            sample_data['model' + str(i)] = solution_mapping_inf(state_var_sol_dict)
        # print(sample_data['model' + str(i)].keys())

    print(sample_data.keys())

    # Create a DataFrame containing the weighted sum of the different model solutions for each variable
    model_weights = dict(zip(sample_data.keys(), weights))
    var_names = sample_data['model0'].keys()
    synth_data_dict = {}
    for vn in var_names:
        this_var = 0 * sample_data['model0'][vn][0]
        for mn in sample_data.keys():
            this_var = this_var + model_weights[mn] * sample_data[mn][vn][0]
        synth_data_dict[vn] = this_var.numpy()
    synth_data_df = pd.DataFrame.from_dict(synth_data_dict)

    # Keep only integer time values so there's one data point per day
    N_t = 100
    t_points = get_tspan(1, N_t, 10 * (N_t - 1) + 1)
    keep_idx = [i for i, t in enumerate(t_points) if int(t) == t]
    synth_data_df = synth_data_df.iloc[keep_idx]

    # Add a date column
    date_col = [start_date] * len(synth_data_df)
    for i in range(len(synth_data_df)):
        date_col[i] = f"{start_date + timedelta(days=i)}"
    synth_data_df["Date"] = date_col
    synth_data_df = synth_data_df.iloc[:, [3, 0, 1, 2]]

    # Reset DataFrame index
    synth_data_df = synth_data_df.reset_index(drop=True)

    return synth_data_df, sample_data


def add_noise(data_df, noise_level, to_plot=True):
    # Function that accepts a DataFrame and level of noise as inputs, and returns and plots the noisy data.
    noisy_data_df = copy.deepcopy(data_df)
    row_num = len(noisy_data_df)
    col_names = ["Cases", "Hospitalizations", "Deaths"]
    col_num = len(col_names)
    noisy_data_df[col_names] = abs(noisy_data_df[col_names] + np.multiply(noise_level * noisy_data_df[col_names],
                                                                          np.random.randn(row_num, col_num)))

    keep_idx = np.arange(len(full_tspan))
    keep_idx = keep_idx[0::10]
    if to_plot:
        for i in range(col_num):
            var_name = col_names[i]
            ax = plot_observations(model1_df[var_name], full_tspan, color="red", label="Model1 " + var_name)
            ax = plot_observations(model2_df[var_name], full_tspan, ax=ax, color="blue", label="Model2 " + var_name)
            ax = plot_observations(model3_df[var_name], full_tspan, ax=ax, color="orange", label="Model3 " + var_name)
            ax = plot_observations(data_df[var_name], full_tspan[keep_idx], ax=ax, color="black",
                                   label="Synth " + var_name)
            ax = plot_observations(noisy_data_df[var_name], full_tspan[keep_idx], ax=ax, color="grey", marker="x",
                                   label="Noisy Synth " + var_name)
            ax.set_title("Synthetic " + var_name[:-1] + " Data")
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("Number of People")

    return noisy_data_df