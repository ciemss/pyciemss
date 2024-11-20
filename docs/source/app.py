from flask import Flask, request, render_template, send_file, redirect, url_for
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
import base64
import pyciemss.visuals.plots as plots
import pyciemss.visuals.trajectories as trajectories
from pyciemss.integration_utils.intervention_builder import (
    param_value_objective,
    start_time_objective,
)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
app = Flask(__name__)
_output_root = None

# Define correct output directory
OUTPUT_BASE_DIR = Path("/Users/oost464/Local_Project/shap/pyciemss/docs/source/output")

MODEL_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/models/"
DATA_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/datasets/"

model1 = os.path.join(MODEL_PATH, "SEIRHD_NPI_Type1_petrinet.json")
model2 = os.path.join(MODEL_PATH, "SEIRHD_NPI_Type2_petrinet.json")
model3 = os.path.join(MODEL_PATH, "SIR_stockflow.json")

dataset1 = os.path.join(DATA_PATH, "SIR_data_case_hosp.csv")
dataset2 = os.path.join(DATA_PATH, "traditional.csv")

models = [model1, model2, model3]  # Example list of models


def save_result(data, name, ref_ext):
    """Save new reference files"""
    file_path = OUTPUT_BASE_DIR / f"{name}.{ref_ext}"
    mode = "w" if ref_ext == "svg" else "wb"
    with open(file_path, mode) as f:
        if ref_ext == "svg":
            json.dump(data, f, indent=4)
        else:
            f.write(data)
    logger.debug(f"Saved result to {file_path}")


def run_simulations(models: Optional[List[str]] = None,
                    interventions: Optional[Dict] = None,
                    calibrate_dataset: Optional[str] = None,
                    ensemble: bool = False):
    global _output_root
    _output_root = OUTPUT_BASE_DIR
    _output_root.mkdir(parents=True, exist_ok=True)

    start_time = 0.0
    end_time = 100.0
    logging_step_size = 10.0
    num_samples = 3

    results = None

    if calibrate_dataset:
        data_mapping = {"case": "infected", "hosp": "hospitalized"}
        num_iterations = 10
        logger.debug(f"Calibrating model with dataset {calibrate_dataset}")
        calibrated_results = pyciemss.calibrate(models[0], calibrate_dataset, data_mapping=data_mapping, num_iterations=num_iterations)
        parameter_estimates = calibrated_results["inferred_parameters"]
        logger.debug("Sampling from calibrated model")
        results = pyciemss.sample(models[0], end_time, logging_step_size, num_samples, start_time=start_time, inferred_parameters=parameter_estimates)

    if interventions:
        logger.debug(f"Sampling from model with interventions: {interventions}")
        results = pyciemss.sample(models[0], end_time, logging_step_size, num_samples, start_time=start_time, static_parameter_interventions=interventions)

    if ensemble:
        solution_mappings = [lambda x: x for _ in models]
        logger.debug(f"Sampling from ensemble of models: {models}")
        results = pyciemss.ensemble_sample(models, solution_mappings, end_time, logging_step_size, num_samples, start_time=start_time)

    if results is None:
        logger.debug(f"Sampling from model: {models[0]}")
        results = pyciemss.sample(models[0], end_time, logging_step_size, num_samples, start_time=start_time)

    nice_labels = {
        "dead_observable_state": "Deceased",
        "hospitalized_observable_state": "Hospitalized",
        "infected_observable_state": "Infected",
    }
    schema = plots.trajectories(pd.DataFrame(results["data"]), keep=["infected_observable_state", "hospitalized_observable_state", "dead_observable_state"], relabel=nice_labels)

    image = plots.ipy_display(schema, format="PNG").data
    logger.debug("Generated image from simulation results")
    save_result(image, "results_schema", "png")
    png_name = "results_schema"
    if calibrate_dataset:
        png_name += "_calibrated"
    if interventions:
        png_name += "_interventions"
    if ensemble:
        png_name += "_ensemble"
    save_result(image, png_name, "png")
    logger.debug(f"Image saved to {_output_root / png_name}.png")
    return str(_output_root / f"{png_name}.png")


@app.route('/', methods=['GET', 'POST'])
def index():
        if request.method == 'POST':
            model_files = request.files.getlist('models')
            models_selected = []
            for model_file in model_files:
                model_path = os.path.join(OUTPUT_BASE_DIR, model_file.filename)
                model_file.save(model_path)
                models_selected.append(model_path)
            calibrate_dataset = request.form.get('calibrate_dataset')
            calibrate_dataset_file = request.files.get('calibrate_dataset')
            if calibrate_dataset_file:
                calibrate_dataset_path = os.path.join(OUTPUT_BASE_DIR, calibrate_dataset_file.filename)
                calibrate_dataset_file.save(calibrate_dataset_path)
                calibrate_dataset = calibrate_dataset_path
            interventions = 'interventions' in request.form
            ensemble = 'ensemble' in request.form
    
            interventions_dict = {torch.tensor(1.): {"gamma": torch.tensor(0.5)}} if interventions else None
    
            png_path = run_simulations(models=models_selected, interventions=interventions_dict if interventions else None,
                           calibrate_dataset=calibrate_dataset if calibrate_dataset else None,
                           ensemble=True if ensemble else False)
            
            with open(png_path, "rb") as image_file:
                image_data = "data:image/png;base64," + base64.b64encode(image_file.read()).decode('utf-8')
        else:
            image_data = None
        return render_template('index.html', models=models, datasets=[dataset1, dataset2], image_data=image_data)


if __name__ == "__main__":
    app.run(debug=True)
