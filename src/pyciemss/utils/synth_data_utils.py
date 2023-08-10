import os
import torch
import pyro
import pyro.distributions as dist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import copy

from datetime import date, timedelta, datetime

from pyro.nn import pyro_method
from pyro.infer import Predictive
from pyro.distributions import Dirichlet
from typing import Dict, Optional

from pyciemss.PetriNetODE.base import (
    ScaledBetaNoisePetriNetODESystem,
    MiraPetriNetODESystem,
    PetriNetODESystem,
    Time,
    State,
    Solution,
    get_name,
)
from pyciemss.utils import state_flux_constraint
from pyciemss.utils import get_tspan
from pyciemss.utils.distributions import ScaledBeta

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


# Plotting utilities
def setup_ax(ax=None):
    if not ax:
        fig = plt.figure(facecolor="w", figsize=(9, 9))
        ax = fig.add_subplot(111, axisbelow=True)

    ax.set_xlabel("Time since start of pandemic (days)")
    ax.set_ylabel("Cases (Prop. of Population)")
    ax.set_yscale("log")
    return ax


def plot_trajectory(prediction, tspan, ax=None, alpha=0.2, color="black", **kwargs):
    tspan = torch.as_tensor(tspan)

    if not ax:
        fig = plt.figure(facecolor="w")
        ax = fig.add_subplot(111, facecolor="#dddddd", axisbelow=True)

    I = prediction["Cases_sol"].detach().numpy().squeeze()
    ax.plot(tspan, I, alpha=0.5, lw=2, color=color, **kwargs)
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)

    return ax


def plot_predictive(
    prediction,
    tspan,
    var_name="Cases_sol",
    ax=None,
    title=None,
    alpha=0.2,
    color="black",
    **kwargs,
):
    tspan = torch.as_tensor(tspan)
    indeces = torch.ones_like(tspan).bool()

    I_low = torch.quantile(prediction[var_name], 0.05, dim=0).detach().numpy()
    I_up = torch.quantile(prediction[var_name], 0.95, dim=0).detach().numpy()

    if not ax:
        fig = plt.figure(facecolor="w")
        ax = fig.add_subplot(111, facecolor="#dddddd", axisbelow=True)

    if title:
        ax.set_title(title)

    ax.fill_between(
        tspan[indeces],
        I_low[indeces],
        I_up[indeces],
        alpha=alpha,
        color=color,
        **kwargs,
    )

    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)

    return ax


def plot_observations(
    data,
    tspan,
    ax=None,
    color="black",
    alpha=0.5,
    lw=0,
    marker=".",
    label=None,
):
    # Plot the data on three separate curves for S(t), I(t) and R(t)

    if not ax:
        fig = plt.figure(facecolor="w")
        ax = fig.add_subplot(111, facecolor="#dddddd", axisbelow=True)

    ax.plot(tspan, data, color, alpha=alpha, lw=lw, marker=marker, label=label)
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)

    return ax


corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
AREA = 0.5 * 1 * 0.75**0.5
triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
refiner = tri.UniformTriRefiner(triangle)
trimesh = refiner.refine_triangulation(subdiv=4)

# For each corner of the triangle, the pair of other corners
pairs = [corners[np.roll(range(3), -i)[1:]] for i in range(3)]
# The area of the triangle formed by point xy and another pair or points
tri_area = lambda xy, pair: 0.5 * np.linalg.norm(np.cross(*(pair - xy)))


def xy2bc(xy, tol=1.0e-4):
    """Converts 2D Cartesian coordinates to barycentric."""
    coords = np.array([tri_area(xy, p) for p in pairs]) / AREA
    return np.clip(coords, tol, 1.0 - tol)


# Adapted from https://blog.bogatron.net/blog/2014/02/02/visualizing-dirichlet-distributions/
def plot_weights(
    samples, ax=None, title=None, concentration=20, nlevels=200, subdiv=7, **kwargs
):
    import math

    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)

    points = torch.tensor(np.array([(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]))
    points /= torch.sum(points, dim=1, keepdim=True)

    alpha = samples * concentration
    vals = torch.stack(
        [
            torch.exp(Dirichlet(alpha).log_prob(points[i, :]))
            for i in range(points.shape[0])
        ]
    )
    vals /= torch.max(vals, dim=0, keepdim=True)[0]
    vals = torch.sum(vals, dim=1)
    vals /= torch.sum(vals)

    if not ax:
        fig = plt.figure(facecolor="w")
        ax = fig.add_subplot(111, facecolor="#dddddd", axisbelow=True)

    if title:
        ax.set_title(title)

    mappable = ax.tricontourf(trimesh, vals, nlevels, cmap="jet", **kwargs)
    ax.axis("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.75**0.5)
    ax.axis("off")

    # label vertices "Model 1", "Model 2", and "Model 3"
    ax.text(-0.05, -0.08, "Model 1", fontsize=14, ha="left")
    ax.text(1.05, -0.08, "Model 2", fontsize=14, ha="right")
    ax.text(0.5, 0.75**0.5 + 0.03, "Model 3", fontsize=14, ha="center")
    # add a colorbar with no numbers
    plt.colorbar(mappable, ticks=[], pad=0.1, ax=ax)
    return ax


def plot_prior_posterior(
    data_df, ensemble_prior_forecasts, all_timepoints, calibrated_solution=None
):
    """Function to plot the prior (and posterior) forecasts with data for comparison"""
    ax = plot_predictive(
        ensemble_prior_forecasts,
        all_timepoints,
        ax=setup_ax(),
        title="Prior Forecasts - Ensemble",
        color="blue",
        label="Ensemble Model Prior Forecasts",
    )
    if calibrated_solution is not None:
        ax = plot_predictive(
            calibrated_solution,
            all_timepoints,
            ax=ax,
            title="Posterior Forecasts - Ensemble",
            color="red",
            label="Ensemble Model Posterior Forecasts",
        )
    ax = plot_observations(
        data_df, all_timepoints, ax=ax, color="black", label="Reported Cases"
    )


def box_plot_weights(generating_weights, calibrated_solution):
    """Function that accepts generating weights and a calibrated ensemble model solution
    as inputs, and makes a box plot of the calibrated ensemble weights to compare
    with the generating weights."""
    fig = plt.figure(facecolor="w", figsize=(9, 9))
    _ = fig.add_subplot(111, axisbelow=True)
    num_models = len(calibrated_solution["model_weights"][0])
    if num_models == 1:
        weight_df = pd.DataFrame(
            {"Model 1 Weight": calibrated_solution["model_weights"][:, 0]}
        )
    elif num_models == 2:
        weight_df = pd.DataFrame(
            {
                "Model 1 Weight": calibrated_solution["model_weights"][:, 0],
                "Model 2 Weight": calibrated_solution["model_weights"][:, 1],
            }
        )
    else:
        weight_df = pd.DataFrame(
            {
                "Model 1 Weight": calibrated_solution["model_weights"][:, 0],
                "Model 2 Weight": calibrated_solution["model_weights"][:, 1],
                "Model 3 Weight": calibrated_solution["model_weights"][:, 2],
            }
        )

    weight_df.boxplot()
    COLOR = ["orange", "blue", "green"]
    for i in range(num_models):
        plt.plot(
            (i + 1),
            generating_weights[i],
            mfc=COLOR[i],
            mec="k",
            ms=7,
            marker="o",
            linestyle="None",
            label="Model " + str(i + 1) + " Generating Weight",
        )
    plt.title("Distribution of Calibrated Ensemble Model Weights")
    plt.legend()


def get_train_test_data(
    data: pd.DataFrame,
    train_start_date: str,
    test_start_date: str,
    test_end_date: str,
    data_total_population: int,
) -> pd.DataFrame:
    """Selects the training and testing data from the dataframe."""

    data_observed_variables = ["Date", "Cases", "Hospitalizations", "Deaths"]
    train_df = data[
        (data[data_observed_variables[0]] >= train_start_date)
        & (data[data_observed_variables[0]] < test_start_date)
    ]
    train_data = [0] * train_df.shape[0]
    start_time = train_df.index[0]

    train_cases = (
        np.array(train_df[data_observed_variables[1]].astype("float"))
        / data_total_population
    )
    train_timepoints = np.array(train_df.index.astype("float"))

    test_df = data[
        (data[data_observed_variables[0]] >= test_start_date)
        & (data[data_observed_variables[0]] < test_end_date)
    ]
    test_cases = (
        np.array(test_df[data_observed_variables[1]].astype("float"))
        / data_total_population
    )
    test_timepoints = np.array(test_df.index.astype("float"))

    for time, row in train_df.iterrows():
        row_dict = {}
        row_dict["Cases"] = row[data_observed_variables[1]] / data_total_population
        row_dict["Deaths"] = row[data_observed_variables[3]] / data_total_population
        if row[data_observed_variables[2]] > 0:
            row_dict["Hospitalizations"] = (
                row[data_observed_variables[2]] / data_total_population
            )

        index = time - start_time
        train_data[index] = (float(time), row_dict)

    all_timepoints = np.concatenate((train_timepoints, test_timepoints))

    return (
        train_data,
        train_cases,
        train_timepoints,
        test_cases,
        test_timepoints,
        all_timepoints,
    )


def create_start_state1(data, start_date, total_pop):
    """Create the start state for Model 1 from data using our best guesses for
    mapping from observed variables to model state variables."""
    start_state = data.set_index("date").loc[start_date].to_dict()

    returned_state = {}
    returned_state["Extinct"] = start_state["cumulative_deaths"]

    # Assume Threatened maps to hospitalizations, and, if no hosp. data, assume 1 hosp. per 100 recorded cases
    if np.isnan(start_state["hospital_census"]):
        returned_state["Threatened"] = start_state["case_census"] / 100
    else:
        returned_state["Threatened"] = start_state["hospital_census"]

    # Assume one undetected case per detected case, and evenly distribute across each category of infectiousness
    returned_state["Diagnosed"] = start_state["case_census"] / 2
    returned_state["Recognized"] = start_state["case_census"] / 2
    returned_state["Infected"] = start_state["case_census"] / 2  # * 10
    returned_state["Ailing"] = (
        start_state["case_census"] / 2
    )  # returned_state["Threatened"] * 100

    # Assume ten individuals are Healed for every one death
    returned_state["Healed"] = start_state["cumulative_deaths"] * 10
    returned_state["Susceptible"] = total_pop - sum(returned_state.values())
    # print(returned_state)
    assert returned_state["Susceptible"] > 0
    return {k: v / total_pop for k, v in returned_state.items()}


def solution_mapping1(model1_solution: dict) -> dict:
    """Create a dictionary of the solution for Model 1, mapping from state variables
    to the observed variables in data."""
    mapped_dict = {}
    mapped_dict["Cases"] = model1_solution["Diagnosed"] + model1_solution["Recognized"]
    mapped_dict["Hospitalizations"] = model1_solution["Threatened"]
    mapped_dict["Deaths"] = model1_solution["Extinct"]
    return mapped_dict


def create_start_state2(data, start_date, m2_total_pop, data_total_pop):
    """Create the start state for Model 2 from data using our best guesses for
    mapping from observed variables in data to model state variables."""
    start_state = data.set_index("date").loc[start_date].to_dict()

    returned_state = {}
    returned_state["Deceased"] = start_state["cumulative_deaths"]

    # If no hospitalization data, assume 1 hospitalized individual per 100 recorded cases
    if np.isnan(start_state["hospital_census"]):
        returned_state["Hospitalized"] = start_state["case_census"] / 100
    else:
        returned_state["Hospitalized"] = start_state["hospital_census"]

    # Assume Infectious maps to recorded cases
    returned_state["Infectious"] = start_state["case_census"]

    # Assume one Asymptomatic (unrecorded) case per recorded case, and that twice as many individuals
    # are Exposed (but not yet infectious) as there are recorded cases
    returned_state["Exposed"] = start_state["case_census"] * 2  # * 20
    returned_state["Asymptomatic"] = start_state["case_census"]  # * 10

    # Assume ten individuals are Recovered for every one death
    returned_state["Recovered"] = start_state["cumulative_deaths"] * 10

    returned_state["Susceptible"] = data_total_pop - sum(returned_state.values())
    assert returned_state["Susceptible"] > 0
    return {k: v * m2_total_pop / data_total_pop for k, v in returned_state.items()}


def solution_mapping2(model2_solution: dict) -> dict:
    """Create a dictionary of the solution for Model 2, mapping from state variables
    to the observed variables in data.
    Note that the population used in Model 2 must be hard-coded here."""
    model2_total_population = 328200000.0
    mapped_dict = {}
    mapped_dict["Cases"] = model2_solution["Infectious"] / model2_total_population
    mapped_dict["Hospitalizations"] = (
        model2_solution["Hospitalized"] / model2_total_population
    )
    mapped_dict["Deaths"] = model2_solution["Deceased"] / model2_total_population
    return mapped_dict


def create_start_state3(data, start_date, m3_total_pop, data_total_pop):
    """Create the start state for Model 3 from data using our best guesses for
    mapping from observed variables in data to model state variables."""
    start_state = data.set_index("date").loc[start_date].to_dict()

    returned_state = {}
    returned_state["Deceased"] = start_state["cumulative_deaths"]

    # Hospitalization rates in model3 are just a scalar multiple of case_census.
    # Initializing here may lead to contradiction for the Infected_reported variable
    # Instead assume initial quarantined individuals are equal to those hospitalized in data
    if np.isnan(start_state["hospital_census"]):
        returned_state["Quarantined"] = start_state["case_census"] * 0.01
    else:
        returned_state["Quarantined"] = start_state["hospital_census"]

    returned_state["Infected_reported"] = start_state["case_census"]

    # Assume one unreported case per recorded case
    returned_state["Infected_unreported"] = start_state["case_census"]

    # Assume twice as many individuals are Exposed (but not yet infectious) as recorded cases
    returned_state["Exposed"] = start_state["case_census"] * 2

    # Assume ten individuals are Recovered for every one death
    returned_state["Recovered"] = start_state["cumulative_deaths"] * 10

    returned_state["Susceptible_confined"] = 0
    returned_state["Susceptible_unconfined"] = data_total_pop - sum(
        returned_state.values()
    )
    assert returned_state["Susceptible_unconfined"] > 0
    return {k: v * m3_total_pop / data_total_pop for k, v in returned_state.items()}


def solution_mapping3(model3_solution: dict) -> dict:
    """Create a dictionary of the solution for Model 3, mapping from state variables
    to the observed variables in data.
    Note that both the population used in Model 3 and the hospitalization ratio must
    be hard-coded here."""
    model3_total_population = 66990210.0
    hosp_ratio = 0.05
    mapped_dict = {}
    mapped_dict["Cases"] = (
        model3_solution["Infected_reported"] / model3_total_population
    )
    mapped_dict["Hospitalizations"] = (
        model3_solution["Infected_reported"] * hosp_ratio
    ) / model3_total_population  # (model3_solution["Infected_unreported"] * hosp_ratio) / model3_total_population
    mapped_dict["Deaths"] = (model3_solution["Deceased"]) / model3_total_population
    return mapped_dict


def get_state_var_names(modelA_sample):
    """Get a list of model state variable names from the sample dictionary."""
    state_var_names = []
    for j in modelA_sample.keys():
        if j[-4:] == "_sol":
            state_var_names.append(j)
    return state_var_names


def plot_intervened_with_data(
    intervened_model,
    model_num,
    time_points,
    sample_dict,
    model_population=None,
    hosp_ratio=None,
):
    model_prior_forecast = sample_petri(intervened_model, time_points, 20)
    model_prior_sol_dict = {}
    state_var_names = get_state_var_names(model_prior_forecast)
    for n in state_var_names:
        model_prior_sol_dict[n[0:-4]] = model_prior_forecast[n]

    if model_num == 1:
        model_solns = solution_mapping1(model_prior_sol_dict)
    elif model_num == 2:
        model_solns = solution_mapping2(model_prior_sol_dict)
    else:
        model_solns = solution_mapping3(model_prior_sol_dict)

    new_model_solns = {f"{k}_sol": v for k, v in model_solns.items()}
    ax = plot_predictive(
        new_model_solns,
        time_points,
        ax=setup_ax(),
        title=f"Intervened Model Output - Model {model_num}",
        color="blue",
        label="Intervened Model Prior Forecasts",
    )
    ax = plot_observations(
        sample_dict["model" + str(model_num)]["Cases"][0],
        time_points,
        ax=ax,
        color="black",
        alpha=0.05,
        label="Intervened Synthetic Cases - Training Data",
    )


def create_synth_data(
    weights,
    start_date,
    t_points,
    data_total_population,
    modelA_sample,
    modelB_sample=None,
    modelC_sample=None,
):
    """Fuction that takes in a set of weights, and returns the synthetic data DataFrame:
    synth_data_df, as well a dictionary sample_data containing the original model output.
    : param weights: ordered list of generating weights used to produce synthetic data
    : param start_date: start date of synthetic data
    : param t_points: time points for which a solution is to be generated
    : param data_total_population: total population
    : param modelA_sample: a single sample from the first model
    : param modelB_sample: a single sample from the second model
    : param modelC_sample: a single sample from the third model
    : return: a DataFrame containing the synthetic data synth_data_df,
    and a dictionary containing the original data from each sample sample_data
    """
    # Function that takes in any number (up to 3) of the previously defined model samples along with a set of weights,
    # and returns the weighted sum of the sample output, aka the synthetic data DataFrame: synth_data_df, as well as the
    # dictionary sample_data containing the original model output.
    # Inputs:
    #       weights = weights!
    # must include one weight per model, weights must be positive and sum to 1

    num_models = len(weights)
    sample_array = [modelA_sample, modelB_sample, modelC_sample]

    # Create synth_data dictionary, a dictionary of dictionaries which has models as keys, and dictionaries
    # containing their outputs (mapped to keys: Cases, Hospitalizations, and Deaths) as values
    sample_data = {}
    for i in range(num_models):
        # Map state variables to data and save output to synth_data dictionary
        state_var_names = get_state_var_names(sample_array[i])
        state_var_sol_dict = {}
        for n in state_var_names:
            state_var_sol_dict[n[0:-4]] = sample_array[i][n]

        if i == 0:
            sample_data["model1"] = solution_mapping1(state_var_sol_dict)
        elif i == 1:
            sample_data["model2"] = solution_mapping2(state_var_sol_dict)
        else:
            sample_data["model3"] = solution_mapping3(state_var_sol_dict)

    # print(sample_data.keys())

    # Create a DataFrame containing the weighted sum of the different model solutions for each variable
    model_weights = dict(zip(sample_data.keys(), weights))
    var_names = sample_data["model1"].keys()
    synth_data_dict = {}
    for vn in var_names:
        this_var = 0 * sample_data["model1"][vn][0]
        for mn in sample_data.keys():
            this_var = this_var + model_weights[mn] * sample_data[mn][vn][0]
        synth_data_dict[vn] = this_var.numpy()
    synth_data_df = pd.DataFrame.from_dict(synth_data_dict)
    synth_data_df = synth_data_df.mul(data_total_population)

    # Keep only integer time values so there's one data point per day
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
    """Function that accepts a DataFrame and level of noise as inputs, and returns
    (and plots) the noisy data.
    : param data_df: DataFrame containing the original data
    : param noise_level: level of noise to add
    : param to_plot: whether to plot the noisy data
    : return: a DataFrame containing the noisy data"""
    noisy_data_df = copy.deepcopy(data_df)
    row_num = len(noisy_data_df)
    col_names = ["Cases", "Hospitalizations", "Deaths"]
    col_num = len(col_names)
    noisy_data_df[col_names] = abs(
        noisy_data_df[col_names]
        + np.multiply(
            noise_level * noisy_data_df[col_names], np.random.randn(row_num, col_num)
        )
    )

    keep_idx = np.arange(len(full_tspan))
    keep_idx = keep_idx[0::10]
    if to_plot:
        for i in range(col_num):
            var_name = col_names[i]
            ax = plot_observations(
                model1_df[var_name], full_tspan, color="red", label="Model1 " + var_name
            )
            ax = plot_observations(
                model2_df[var_name],
                full_tspan,
                ax=ax,
                color="blue",
                label="Model2 " + var_name,
            )
            ax = plot_observations(
                model3_df[var_name],
                full_tspan,
                ax=ax,
                color="orange",
                label="Model3 " + var_name,
            )
            ax = plot_observations(
                data_df[var_name],
                full_tspan[keep_idx],
                ax=ax,
                color="black",
                label="Synth " + var_name,
            )
            ax = plot_observations(
                noisy_data_df[var_name],
                full_tspan[keep_idx],
                ax=ax,
                color="grey",
                marker="x",
                label="Noisy Synth " + var_name,
            )
            ax.set_title("Synthetic " + var_name[:-1] + " Data")
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("Number of People")

    return noisy_data_df


def ensemble_calibration(
    models,
    synth_data_df,
    start_states,
    num_samples,
    num_iterations,
    ensemble_weights,
    train_start_date,
    test_start_date,
    test_end_date,
    data_total_population,
):
    """Ensembles the given models and calibrates them to the synthetic data.
    : param models: ordered list of previously set up/intervened models to be ensembled
    : param synth_data_df: DataFrame containing the synthetic data
    : param start_states: ordered list of start states
    : param num_samples: number of samples to take from each model
    : param num_iterations: number of iterations to perform during calibration
    : param ensemble_weights: list of weights to use as initial guess in ensemble calibration
    : param train_start_date: start date for training
    : param test_start_date: start date for testing
    : param test_end_date: end date for testing
    : param data_total_population: total population
    : return: ensemble_prior_forecasts: distribution of ensemble forecasts before calibration
    : return: ensemble_forecasts: distribution of ensemble forecases after calibration
    : return: all_timepoints: list of all timepoints (for plotting)"""

    (
        train_data,
        train_cases,
        train_timepoints,
        test_cases,
        test_timepoints,
        all_timepoints,
    ) = get_train_test_data(
        synth_data_df,
        train_start_date,
        test_start_date,
        test_end_date,
        data_total_population,
    )
    start_time = train_timepoints[0] - 1e-5

    # Set up the ensemble model
    ensemble_total_population = 1.0
    dirichlet_concentration = 1.0
    noise_pseudocount = 100.0
    solution_mappings = [solution_mapping1, solution_mapping2, solution_mapping3]
    ensemble = setup_model_ensemble(
        models,
        ensemble_weights,
        solution_mappings,
        start_time,
        start_states,
        ensemble_total_population,
        dirichlet_concentration=dirichlet_concentration,
        noise_pseudocount=noise_pseudocount,
    )
    display(ensemble)

    # Sample from the ensemble prior
    ensemble_prior_forecasts = sample_ensemble(ensemble, all_timepoints, num_samples)

    # Calibrate the ensemble model to the synthetic data
    autoguide = pyro.infer.autoguide.AutoDiagonalNormal
    inferred_parameters = calibrate_ensemble(
        ensemble,
        train_data,
        num_iterations=num_iterations,
        verbose=True,
        verbose_every=10,
        autoguide=autoguide,
    )
    calibrated_solution = sample_ensemble(
        ensemble,
        timepoints=all_timepoints,
        num_samples=50,
        inferred_parameters=inferred_parameters,
    )

    return ensemble_prior_forecasts, calibrated_solution, all_timepoints
