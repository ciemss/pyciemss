import os
import pyciemss
import shap
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from functools import lru_cache
from collections import defaultdict
import logging
plt.rcParams.update({'font.size': 14})

# Setup logging
current_time = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
log_filename = os.path.join("output", f'log_{current_time}.txt')
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')

start_time = 0.0
end_time = 30
model1 = "SEIRHD_NPI_Type1_petrinet.json"
logging_step_size = 10.0
num_samples = 10
timepoint_focus = 2
focus = 'H_state'
intervention_values = {
    "beta_c": np.linspace(0, 3.0, 2),
    "gamma": np.linspace(0, 3.0, 2)
}

results_filename = os.path.join(os.path.dirname(__file__), f'simulation_results_{current_time}.csv')
# Define the filename for saving the SHAP summary plot
plot_filename = os.path.join("output", f"shap_summary_plot_samples_{num_samples}_combinations.png")
plot_path = os.path.join(os.path.dirname(__file__), plot_filename)

param_call_count = defaultdict(int)

@lru_cache(maxsize=None)
def run_simulation(intervention_tuple):
    logging.info(f"Running simulation for: {intervention_tuple}")
    intervention = {
        torch.tensor(start_time): {
            name: torch.tensor(value)
            for name, value in intervention_tuple
        }
    }
    result = pyciemss.sample(model1, end_time, logging_step_size, num_samples, start_time=start_time,
                             static_parameter_interventions=intervention)
    return result
    

def save_results_to_csv(input_row, median_value):
    if not os.path.exists(results_filename):
        results_df = pd.DataFrame(columns=list(input_row.index) + ['result'])
        results_df.to_csv(results_filename, index=False)
    
    current_result = input_row.to_dict()
    current_result['result'] = median_value
    results_df = pd.DataFrame([current_result])
    results_df.to_csv(results_filename, mode='a', header=False, index=False)

def sim(inputs):
    results = []
    for _, input_row in inputs.iterrows():
        intervention_tuple = tuple(sorted(input_row.items()))
        param_call_count[intervention_tuple] += 1
        logging.info(f"Calling sim with intervention_tuple: {intervention_tuple}")

        result = run_simulation(intervention_tuple)
        filtered_results = result["data"][result["data"]["timepoint_id"] == timepoint_focus]
        median_value = np.median(filtered_results[focus])
        results.append(median_value)

        save_results_to_csv(input_row, median_value)

    logging.info(f"Parameter Call Count: {dict(param_call_count)}")
    return np.array(results)

def plot_histograms(inputs, results, min_params, max_params, top_5_percent_params):

    # Plot histogram of the results before filtering
    plt.figure(figsize=(10, 6))
    plt.hist(results, bins=10, alpha=0.7, label='Results')
    plt.axvline(np.percentile(results, 95), color='r', linestyle='dashed', linewidth=1, label='95th Percentile')
    plt.xlabel('Results')
    plt.ylabel('Frequency')
    plt.title('Histogram of Results Before Filtering')
    plt.legend()
    plt.savefig(os.path.join("output", f"results_histogram_{current_time}.png"))
    plt.close()

    # Plot histogram of the parameters before filtering
    for param in intervention_values.keys():
        plt.figure(figsize=(10, 6))
        hist_data = inputs[param]
        if hist_data.min() == hist_data.max():
            hist_data = hist_data + np.random.normal(0, 0.01, size=hist_data.shape)
        plt.hist(hist_data, bins=10, alpha=0.7, label=f'{param} values')
        plt.axvline(min_params[param], color='g', linestyle='dashed', linewidth=3, label='Min Param (Top 5%)')
        plt.axvline(max_params[param], color='r', linestyle='dashed', linewidth=3, label='Max Param (Top 5%)')
        plt.xlabel(param)
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {param} Before Filtering')
        plt.legend()
        plt.savefig(os.path.join("output", f"{param}_histogram_{current_time}.png"))
        plt.close()

  
    # Create scatter plots for each combination of parameters in the top 5% parameters
    num_params = len(intervention_values.keys())
    fig, axes = plt.subplots(num_params, num_params, figsize=(15, 15))
    fig.suptitle('Scatter Plots of Top 5% Parameter Combinations')
    for i, param1 in enumerate(intervention_values.keys()):
        for j, param2 in enumerate(intervention_values.keys()):
            if i < j:
                axes[i, j].scatter(top_5_percent_params[param1], top_5_percent_params[param2], alpha=0.7)
                axes[i, j].set_xlabel(param1)
                axes[i, j].set_ylabel(param2)
            else:
                axes[i, j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    scatter_plot_filename = os.path.join("output", f"scatter_plots_top_5_percent_params_{current_time}.png")
    plt.savefig(scatter_plot_filename)
    plt.close()


def get_extreme_values(inputs):
    results = sim(inputs)
    threshold = np.percentile(results, 95)
    top_5_percent_indices = np.where(results >= threshold)[0]
    
    top_5_percent_params = inputs.iloc[top_5_percent_indices]
    min_params = top_5_percent_params.min()
    max_params = top_5_percent_params.max()

    logging.info(f"Top 5% value params: {top_5_percent_params}")
    logging.info(f"Minimum value params: {min_params}")
    logging.info(f"Maximum value params: {max_params}")
    plot_histograms(inputs, results, min_params, max_params, top_5_percent_params)
    return min_params, max_params

parameters_df = pd.DataFrame([
    dict(zip(intervention_values.keys(), values))
    for values in product(*intervention_values.values())
])

# Call the function to get parameters with extreme values
min_params, max_params = get_extreme_values(parameters_df)

# Use the min_params to set new beta_values and gamma_values
beta_min = min_params['beta_c']
beta_max = max_params['beta_c'] if min_params['beta_c'] != max_params['beta_c'] else min_params['beta_c'] + 0.2
gamma_min = min_params['gamma']
gamma_max = max_params['gamma'] if min_params['gamma'] != max_params['gamma'] else min_params['gamma'] + 0.2
print(f"New beta_min: {beta_min}, beta_max: {beta_max}, gamma_min: {gamma_min}, gamma_max: {gamma_max}")
intervention_values["beta_c"] = np.linspace(beta_min, beta_max, 10)
intervention_values["gamma"] = np.linspace(gamma_min, gamma_max, 10)

# Create a DataFrame directly from the product of intervention values
parameters_df = pd.DataFrame([
    dict(zip(intervention_values.keys(), values))
    for values in product(*intervention_values.values())
])

explainer = shap.Explainer(sim, parameters_df)
shap_values = explainer(parameters_df)

# Convert SHAP values to a DataFrame
shap_values_df = pd.DataFrame(shap_values.values, columns=intervention_values.keys())
shap_values_filename = os.path.join(os.path.dirname(__file__), os.path.join("output", f'shap_values_{current_time}.csv'))
shap_values_df.to_csv(shap_values_filename, index=False)

# Create the SHAP waterfall plot for the first sample and save it
plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_values[0], show=False)
waterfall_plot_filename = os.path.join("output", f"shap_waterfall_plot_{current_time}.png")
plt.savefig(waterfall_plot_filename)
plt.close()


# Create the SHAP summary plot and save it
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, feature_names=list(intervention_values.keys()), show=False)
plt.savefig(plot_path)
plt.close()
