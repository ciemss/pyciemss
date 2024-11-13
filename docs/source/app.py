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
import pyciemss.visuals.plots as plots
import pyciemss.visuals.trajectories as trajectories
from pyciemss.integration_utils.intervention_builder import (
    param_value_objective,
    start_time_objective,
)


app = Flask(__name__)
_output_root = None

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

    if output_dir is None:
        output_dir = f"output"
    _output_root = Path(output_dir)
    _output_root.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.info(f"Output will be saved to {_output_root}")

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
    save_result(image, "results_schema", "png")
    png_name = "results_schema"
    if calibrate_dataset:
        png_name += "_calibrated"
    if interventions:
        png_name += "_interventions"
    if ensemble:
        png_name += "_ensemble"
    save_result(image, png_name, "png")
    return png_name


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        models_selected = request.form.getlist('models')
        calibrate_dataset = request.form.get('calibrate_dataset')
        interventions = request.form.get('interventions')
        ensemble = request.form.get('ensemble')

        interventions_dict = {torch.tensor(1.): {"gamma": torch.tensor(0.5)}} if interventions else None

        png_name = run_simulations(models=models_selected, interventions=interventions_dict if interventions else None,
                                   calibrate_dataset=calibrate_dataset if calibrate_dataset else None,
                                   ensemble=True if ensemble else False)
        return redirect(url_for('show_image', image_name=png_name))

    return render_template('index.html', models=models, datasets=[dataset1, dataset2])


@app.route('/image/<image_name>')
def show_image(image_name):
    image_path = _output_root / f"{image_name}.png"
    return send_file(image_path, mimetype='image/png')


if __name__ == "__main__":
    app.run(debug=True)
