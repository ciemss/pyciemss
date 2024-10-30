import os
import pyciemss
import shap
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

MODEL_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/models/"
model1 = os.path.join(MODEL_PATH, "SEIRHD_NPI_Type1_petrinet.json")

# Define simulation sample parameters
start_time = 0.0
end_time = 100
logging_step_size = 10.0
num_samples = 5
timepoint_focus = 10

# Define the intervention names and generate plausible random/step values for inputs
intervention_names = ["beta_c", "gamma", "total_population"]
beta_values = np.linspace(0.1, 1.0, 2)
gamma_values = np.linspace(0.1, 1.0, 2)
total_population_values = np.linspace(19340000, 29340000, 2)

# Create input combinations as dictionaries
input_combinations = list(product(beta_values, gamma_values, total_population_values))

# Convert input combinations to DataFrame
parameters_df = pd.DataFrame([
    {name: value for name, value in zip(intervention_names, values)}
    for values in input_combinations
])


# Run pysimss simulation with interventions and return median values of the focus variable
def sim(inputs, focus='H_state', timepoint_focus = 10):
    """
    Simulate the model with given inputs and return the median values of the focus variable at a specific timepoint.
    Parameters:
    inputs (pd.DataFrame): A DataFrame where each row represents a set of input parameters for the simulation.
    focus (str): The variable to focus on for extracting median values from the simulation results. Default is 'H_state'.
    timepoint_focus (int): The specific timepoint to filter the simulation results. Default is 10.
    Returns:
    np.ndarray: An array of median values of the focus variable at the specified timepoint for each set of inputs.
    """
    results = []
    for _, input_row in inputs.iterrows():
        # Create intervention based on input row series
        intervention = {
            torch.tensor(start_time): {
                name: torch.tensor(value)
                for name, value in input_row.items()
            }
        }
        
        # Run the simulation with the intervention
        result = pyciemss.sample(model1, end_time, logging_step_size, num_samples, start_time=start_time,
                                 static_parameter_interventions=intervention)

        # Filter the results by the desired timepoint
        filtered_results = result["data"][result["data"]["timepoint_id"] == timepoint_focus]
        
        # Get the median value of the focus variable from the filtered results
        median_value = np.median(filtered_results[focus])
        results.append(median_value)
    return np.array(results)

# Use SHAP to explain the model with the sim function
def sim_wrapper(df):
    return sim(df, focus='H_state', timepoint_focus = 10)

explainer = shap.Explainer(sim_wrapper, parameters_df)  
shap_values = explainer(parameters_df)

# Plot SHAP values
plt.figure(figsize=(10, 6))
num_combinations = len(beta_values) * len(gamma_values)
shap.summary_plot(shap_values, feature_names=intervention_names)
plot_filename = f"shap_summary_plot_samples_{num_samples}_combinations_{num_combinations}.png"
plot_path = os.path.join(os.path.dirname(__file__), plot_filename)
plt.savefig(plot_path)
