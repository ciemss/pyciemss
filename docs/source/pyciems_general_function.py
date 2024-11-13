#!/usr/bin/env python
# coding: utf-8

import os
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

   # ### Select models and data

MODEL_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/models/"
DATA_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/datasets/"

model1 = os.path.join(MODEL_PATH, "SEIRHD_NPI_Type1_petrinet.json")
model2 = os.path.join(MODEL_PATH, "SEIRHD_NPI_Type2_petrinet.json")
model3 = os.path.join(MODEL_PATH, "SIR_stockflow.json")

dataset1 = os.path.join(DATA_PATH, "SIR_data_case_hosp.csv")
dataset2 = os.path.join(DATA_PATH, "traditional.csv")

_output_root = None

def save_result(data, name, ref_ext):
    """Save new reference files"""
    _output_root.mkdir(parents=True, exist_ok=True)
    mode = "w" if ref_ext == "svg" else "wb"
    with open(_output_root / f"{name}.{ref_ext}", mode) as f:
        f.write(data)

def run_simulations(output_dir: Optional[str] = None, 
                    interventions: Optional[Dict] = None, 
                    calibrate_dataset: Optional[str] = None):
    global _output_root

    # Configure output directory
    if output_dir is None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"output_{current_time}"
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

    # ## Sample interface
    # Take `num_samples` number of samples from the (prior) distribution invoked by the chosen model.

    # ### Sample from model 1

    logger.debug(f"Sampling from model 1: {model1}")
    result1 = pyciemss.sample(model1, end_time, logging_step_size, num_samples, start_time=start_time)

    # ### Sample from model 2

    logger.debug(f"Sampling from model 2: {model2}")
    result2 = pyciemss.sample(model2, end_time, logging_step_size, num_samples, start_time=start_time)

    # ## Ensemble Sample Interface
    # Sample from an ensemble of model 1 and model 2 

    model_paths = [model1, model2]
    solution_mappings = [lambda x : x, lambda x : x] # Conveniently, these two models operate on exactly the same state space, with the same names.

    logger.debug(f"Sampling from ensemble of models: {model_paths}")
    ensemble_result = pyciemss.ensemble_sample(model_paths, solution_mappings, end_time, logging_step_size, num_samples, start_time=start_time)

    # ## Calibrate model if dataset is provided
    if calibrate_dataset:
        data_mapping = {"case": "infected", "hosp": "hospitalized"} # data is mapped to observables
        num_iterations = 10
        logger.debug(f"Calibrating model 1 with dataset {calibrate_dataset}")
        calibrated_results = pyciemss.calibrate(model1, calibrate_dataset, data_mapping=data_mapping, num_iterations=num_iterations)
        parameter_estimates = calibrated_results["inferred_parameters"]

        # Use calibrated parameter estimates in `sample` to sample from the calibrated model (posterior distribution).
        logger.debug("Sampling from calibrated model 1")
        calibrated_sample_results = pyciemss.sample(model1, end_time, logging_step_size, num_samples, 
                                                    start_time=start_time, inferred_parameters=parameter_estimates)

    # ## Sample interface with interventions if provided
    if interventions:
        logger.debug(f"Sampling from model with interventions: {interventions}")
        result = pyciemss.sample(model3, end_time, logging_step_size, num_samples, start_time=start_time, static_parameter_interventions=interventions)

    # Plot the result one time at the end of all operations
    # Plot the result
    nice_labels = {
        "dead_observable_state": "Deceased", 
        "hospitalized_observable_state": "Hospitalized",
        "infected_observable_state": "Infected",
    }
    schema = plots.trajectories(pd.DataFrame(calibrated_sample_results["data"]), 
                               keep=["infected_observable_state", "hospitalized_observable_state", "dead_observable_state"], 
                               relabel=nice_labels)
    save_result(schema, "schema_output", "svg")
    logger.info(f"Saved schema to schema_output.svg")




# Example usage
if __name__ == "__main__":
    # Example static parameter intervention
    example_interventions = {torch.tensor(1.): {"p_cbeta": torch.tensor(0.5)}}
    
    # Run simulations without interventions and without calibration dataset
    run_simulations()

    # Run simulations with only interventions
    run_simulations(interventions=example_interventions)

    # Run simulations with only calibration dataset
    run_simulations(calibrate_dataset=dataset1)
