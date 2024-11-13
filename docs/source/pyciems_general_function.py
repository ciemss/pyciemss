#!/usr/bin/env python
# coding: utf-8

import os
import json
import pyciemss
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime
from pathlib import Path

import pyciemss.visuals.plots as plots
import pyciemss.visuals.trajectories as trajectories

from pyciemss.integration_utils.intervention_builder import (
    param_value_objective,
    start_time_objective,
)

_output_root = None

def save_result(data, name, ref_ext):
    """Save new reference files"""
    _output_root.mkdir(parents=True, exist_ok=True)
    mode = "w" if ref_ext == "svg" else "wb"
    with open(_output_root / f"{name}.{ref_ext}", mode) as f:
        if ref_ext == "svg":
            json.dump(data, f, indent=4)
        else:
            f.write(data)

def run_simulations(output_dir: Optional[str] = None,
                    models: Optional[List[str]] = None, 
                    interventions: Optional[Dict] = None, 
                    calibrate_dataset: Optional[str] = None,
                    ensemble: bool = False):    
    global _output_root

    # Configure output directory
    if output_dir is None:
        #current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"output"
    _output_root = Path(output_dir)
    _output_root.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.info(f"Output will be saved to {_output_root}")

    # ### Set parameters for sampling

    start_time = 0.0
    end_time = 100.0
    logging_step_size = 10.0
    num_samples = 3

    results = None

    if calibrate_dataset:
        data_mapping = {"case": "infected", "hosp": "hospitalized"} # data is mapped to observables
        num_iterations = 10
        logger.debug(f"Calibrating model with dataset {calibrate_dataset}")
        calibrated_results = pyciemss.calibrate(models[0], calibrate_dataset, data_mapping=data_mapping, num_iterations=num_iterations)
        parameter_estimates = calibrated_results["inferred_parameters"]

        # Use calibrated parameter estimates in `sample` to sample from the calibrated model (posterior distribution).
        logger.debug("Sampling from calibrated model")
        results = pyciemss.sample(models[0], end_time, logging_step_size, num_samples, 
                                  start_time=start_time, inferred_parameters=parameter_estimates)

    if interventions:
        logger.debug(f"Sampling from model with interventions: {interventions}")
        results = pyciemss.sample(models[0], end_time, logging_step_size, num_samples, 
                                  start_time=start_time, static_parameter_interventions=interventions)

    if ensemble:
        solution_mappings = [lambda x: x for _ in models]
        logger.debug(f"Sampling from ensemble of models: {models}")
        results = pyciemss.ensemble_sample(models, solution_mappings, end_time, logging_step_size, num_samples, start_time=start_time)
    
    # If no special operations, sample from models normally
    if results is None:
        logger.debug(f"Sampling from model: {models[0]}")
        results = pyciemss.sample(models[0], end_time, logging_step_size, num_samples, start_time=start_time)

    # Plot the results
    nice_labels = {
        "dead_observable_state": "Deceased", 
        "hospitalized_observable_state": "Hospitalized",
        "infected_observable_state": "Infected",
    }
    schema = plots.trajectories(pd.DataFrame(results["data"]), 
                               keep=["infected_observable_state", "hospitalized_observable_state", "dead_observable_state"], 
                               relabel=nice_labels)
    # save_result(schema, "results_schema", "svg")

    image = plots.ipy_display(schema, format="PNG").data
    save_result(image, "results_schema", "png")
    png_name = "results_schema"
    if calibrate_dataset:
        png_name += "_calibrated"
    if interventions:
        png_name += "_interventions"
    if ensemble:
        png_name += "_ensemble"
    save_result(image, png_name, "png")
    logger.info(f"Saved schema to results_schema.svg")


# Example models and datasets
MODEL_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/models/"
DATA_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/datasets/"

model1 = os.path.join(MODEL_PATH, "SEIRHD_NPI_Type1_petrinet.json")
model2 = os.path.join(MODEL_PATH, "SEIRHD_NPI_Type2_petrinet.json")
model3 = os.path.join(MODEL_PATH, "SIR_stockflow.json")

dataset1 = os.path.join(DATA_PATH, "SIR_data_case_hosp.csv")
dataset2 = os.path.join(DATA_PATH, "traditional.csv")

models = [model1, model2, model3]  # Example list of models

# Example usage demonstrating various scenarios
if __name__ == "__main__":
    # Example intervention
    example_interventions = {torch.tensor(1.): {"gamma": torch.tensor(0.5)}}
    
    # Example 1: Using the first model and an intervention
    run_simulations(models=[model1], interventions=example_interventions)

    # Example 2: Using the first model and calibration dataset
    run_simulations(models=[model1], calibrate_dataset=dataset1)

    # # Example 3: Using the first model, intervention, and calibration dataset
    run_simulations(models=[model1], interventions=example_interventions, calibrate_dataset=dataset1)

    # # Example 4: Using an ensemble of models
    # run_simulations(models=models, ensemble=True)
