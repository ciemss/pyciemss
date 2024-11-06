import os
import pyciemss
import shap
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from functools import lru_cache

start_time = 0.0
end_time = 30
model1 = "SEIRHD_NPI_Type1_petrinet.json"
logging_step_size = 10.0
num_samples = 10
timepoint_focus = 2
focus='H_state'
intervention_names = ["beta_c", "gamma"]
beta_values = np.linspace(0.1, 2.0, 10)
gamma_values = np.linspace(0.1, 2.0, 10)

input_combinations = list(product(beta_values, gamma_values))
parameters_df = pd.DataFrame([
    {name: value for name, value in zip(intervention_names, values)}
    for values in input_combinations
])

current_time = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
results_filename = os.path.join(os.path.dirname(__file__), f'simulation_results_{current_time}.csv')

@lru_cache(maxsize=None)
def run_simulation(intervention_tuple):
    intervention = {
        torch.tensor(start_time): {
            name: torch.tensor(value)
            for name, value in intervention_tuple
        }
    }
    result = pyciemss.sample(model1, end_time, logging_step_size, num_samples, start_time=start_time,
                             static_parameter_interventions=intervention)
    return result

def sim(inputs):
    results = []
    for _, input_row in inputs.iterrows():
        intervention_tuple = tuple(sorted(input_row.items()))

        print(f"Calling sim with intervention_tuple: {intervention_tuple}")
        result = run_simulation(intervention_tuple)
        filtered_results = result["data"][result["data"]["timepoint_id"] == timepoint_focus]
        median_value = np.median(filtered_results[focus])
        
        if not os.path.exists(results_filename):
            results_df = pd.DataFrame(columns=list(input_row.index) + ['result'])
            results_df.to_csv(results_filename, index=False)
        
        current_result = input_row.to_dict()
        current_result['result'] = median_value
        results_df = pd.DataFrame([current_result])
        results_df.to_csv(results_filename, mode='a', header=False, index=False)
        results.append(median_value)

    return np.array(results)

explainer = shap.Explainer(sim, parameters_df)  
shap_values = explainer(parameters_df)
results_df = pd.read_csv(results_filename)

# Convert SHAP values to a DataFrame
shap_values_df = pd.DataFrame(shap_values.values, columns=intervention_names)

# Define the filename for saving SHAP values
shap_values_filename = os.path.join(os.path.dirname(__file__), os.path.join("output", f'shap_values_{current_time}.csv'))

# Save SHAP values to a CSV file
shap_values_df.to_csv(shap_values_filename, index=False)

# Define the filename for saving the SHAP summary plot
plot_filename = os.path.join("output", f"shap_summary_plot_samples_{num_samples}_combinations.png")
plot_path = os.path.join(os.path.dirname(__file__), plot_filename)

# Create the SHAP summary plot and save it
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, feature_names=intervention_names, show=False)
plt.savefig(plot_path)
plt.close()
