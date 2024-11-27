from flask import Flask, request, render_template, send_file, redirect, url_for, jsonify,  send_from_directory
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
                    calibrate_dataset: Optional[str] = None, keep: List[str] = None):
    global _output_root
    _output_root = OUTPUT_BASE_DIR
    _output_root.mkdir(parents=True, exist_ok=True)

    start_time = 0.0
    end_time = 30.0
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


    if results is None:
        logger.debug(f"Sampling from model: {models[0]}")
        results = pyciemss.sample(models[0], end_time, logging_step_size, num_samples, start_time=start_time)


    print(pd.DataFrame(results["data"]).columns)
    print(keep)
    schema = plots.trajectories(pd.DataFrame(results["data"]))

    try:
        image = plots.ipy_display(schema, format="PNG").data
    except ValueError as e:
        logger.error(f"Vega to PNG conversion failed: {e}")
        return None
    logger.debug("Generated image from simulation results")
    save_result(image, "results_schema", "png")
    png_name = "results_schema"
    save_result(image, png_name, "png")
    logger.debug(f"Image saved to {_output_root / png_name}.png")
        # Save the image to a file

    # Save the image to a file
    image_path = os.path.join(OUTPUT_BASE_DIR, "results_schema.png")
    with open(image_path, "wb") as f:
        f.write(image)
    return "results_schema.png"

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
        calibrate_dataset_file = request.files.get('calibrate_dataset_file')
        if calibrate_dataset_file:
            calibrate_dataset_path = os.path.join(OUTPUT_BASE_DIR, calibrate_dataset_file.filename)
            calibrate_dataset_file.save(calibrate_dataset_path)
            calibrate_dataset = calibrate_dataset_path

        observables = request.form.getlist('observables')
        observables = [obs + "_observable_state" for obs in observables]
        
        interventions = 'interventions' in request.form

        # Extract parameter IDs and values from the form
        param_ids = request.form.getlist('param_id')
        param_values = request.form.getlist('param_value')
        param_start_times = request.form.getlist('param_start_time')

        param_ids_2 = request.form.getlist('param_id_2')
        param_values_2 = request.form.getlist('param_value_2')
        param_start_times_2 = request.form.getlist('param_start_time_2')

        param_ids_3 = request.form.getlist('param_id_3')
        param_values_3 = request.form.getlist('param_value_3')
        param_start_times_3 = request.form.getlist('param_start_time_3')

        # Construct the interventions_dict
        interventions_dict = {}
        if interventions:
            for param_id, param_value, param_start_time in zip(param_ids, param_values, param_start_times):
                if param_id and param_value and param_start_time:
                    interventions_dict[torch.tensor(float(param_start_time))] = {param_id: torch.tensor(float(param_value))}
            for param_id, param_value, param_start_time in zip(param_ids_2, param_values_2, param_start_times_2):
                if param_id and param_value and param_start_time:
                    interventions_dict[torch.tensor(float(param_start_time))] = {param_id: torch.tensor(float(param_value))}
            for param_id, param_value, param_start_time in zip(param_ids_3, param_values_3, param_start_times_3):
                if param_id and param_value and param_start_time:
                    interventions_dict[torch.tensor(float(param_start_time))] = {param_id: torch.tensor(float(param_value))}
        print("Simulations.")
        png_filename = run_simulations(
            models=models_selected,
            keep=observables,
            interventions=interventions_dict if interventions else None,
            calibrate_dataset=calibrate_dataset if calibrate_dataset else None
        )
        
        return render_template('index.html', models=models_selected, datasets=[calibrate_dataset], image_filename=png_filename)

    return render_template('index.html', models=[], datasets=[], image_filename=None)

@app.route('/get_image/<filename>')
def get_image(filename):
    return send_from_directory(OUTPUT_BASE_DIR, filename)
@app.route('/get_model_info', methods=['POST'])
def get_model_info():
    models_file = request.files.get('models_file')
    if models_file:
        try:
            models_path = os.path.join(OUTPUT_BASE_DIR, models_file.filename)
            models_file.save(models_path)
            with open(models_path, 'r') as file:
                data = json.load(file)
            
            # Extract model states
            model_states = [state['id'] for state in data.get('model', {}).get('stocks', [])]
            
            # Extract observable names
            observables = [obs['name'] for obs in data.get('semantics', {}).get('ode', {}).get('observables', [])]
            
            return jsonify({'model_states': model_states, 'observables': observables})
        except Exception as e:
            app.logger.error(f"Error processing file: {e}")
            return jsonify({'error': str(e)}), 500
    return jsonify({'model_states': [], 'observables': []}), 400

@app.route('/update_max_timestamp', methods=['POST'])
def update_max_timestamp():
    calibrate_dataset_file = request.files.get('calibrate_dataset_file')
    if calibrate_dataset_file:
        try:
            calibrate_dataset_path = os.path.join(OUTPUT_BASE_DIR, calibrate_dataset_file.filename)
            calibrate_dataset_file.save(calibrate_dataset_path)
            df = pd.read_csv(calibrate_dataset_path)
            max_timestamp = df['Timestamp'].max()
            app.logger.debug(f"Largest timestamp from calibrate_dataset: {max_timestamp}")
            return jsonify({'max_timestamp': max_timestamp})
        except Exception as e:
            app.logger.error(f"Error processing file: {e}")
            return jsonify({'error': str(e)}), 500
    return jsonify({'max_timestamp': None}), 400

@app.route('/get_parameters', methods=['POST'])
def get_parameters():
    print("Getting parameters")
    models_file = request.files.get('models_file')
    if models_file:
        try:
            models_path = os.path.join(OUTPUT_BASE_DIR, models_file.filename)
            models_file.save(models_path)
            with open(models_path, 'r') as file:
                data = json.load(file)
            parameters = data.get('semantics', {}).get('ode', {}).get('parameters', [])
            if not isinstance(parameters, list):
                parameters = []
            app.logger.debug(f"Extracted parameters: {json.dumps(parameters, indent=4)}")
            extracted_parameters = [
                {'id': param.get('id'), 'value': param.get('value')}
                for param in parameters
            ]
            return jsonify(extracted_parameters)
        except Exception as e:
            app.logger.error(f"Error processing file: {e}")
            return jsonify({'error': str(e)}), 500
    return jsonify([]), 400

if __name__ == "__main__":
    app.run(debug=True)
