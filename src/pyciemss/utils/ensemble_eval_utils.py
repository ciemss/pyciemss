# Load dependencies
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyciemss.Ensemble.interfaces import load_and_sample_petri_ensemble, load_and_calibrate_and_sample_ensemble_model
from pyciemss.PetriNetODE.interfaces import load_and_sample_petri_model
from pyciemss.visuals import plots
from pyciemss.utils.interface_utils import cdc_reformatcsv, make_quantiles, convert_to_output_format

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def data_selector_function(data_df, train_start_row_num, train_end_row_num, forecast_end_row_num):
    '''
    This function produces two datasets: all_data and train_data, and a list timepoints for simulation
    '''
    all_data = data_df[train_start_row_num:forecast_end_row_num].reset_index()
    all_data = all_data.drop(columns="timestep")
    all_data = all_data.drop(columns="index")

    train_data = data_df[train_start_row_num:train_end_row_num].reset_index(drop=True)
    train_data1 = train_data.assign(timepoints=[float(i) for i in range(len(train_data))])
    train_data = train_data1[["I", "E", "H", "D", "I0", "I1", "I2", "I3", "H0", "H1", "H2", "H3"]]

    num_timepoints = (len(all_data) - 1)*10 + 1
    simulation_timepoints = [round(i * 0.1, 1) for i in range(0, num_timepoints)]
    
    return all_data, train_data, simulation_timepoints

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def plot_case_hosp_death_data(N, data, forecast1_start, forecast2_start, forecast3_start, forecast4_start, per100K=True):
    '''
    Plot case, hospitalization, and death data
    '''
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    
    if per100K:
        ax1.scatter(data.index, 100000*(data.I/N), color="red")
        ax1.set_title("Cases per 100,000")
    else:
        ax1.scatter(data.index, data.I, color="red")
        ax1.set_title("Cases")
    ax1.axvline(x = forecast1_start, color = 'darkgreen', linestyle ="--", label = 'begin forecast 1')
    ax1.axvline(x = forecast2_start, color = 'forestgreen', linestyle =":", label = 'begin forecast 2')
    ax1.axvline(x = forecast3_start, color = 'green', linestyle ="-.", label = 'begin forecast 3')
    ax1.axvline(x = forecast4_start, color = 'teal', label = 'begin forecast 4')
    ax1.legend(loc='lower right')

    if per100K:
        ax2.scatter(data.index, 100000*(data.H/N), color="navy")
        ax2.set_title("Hospitalizations per 100,000")
    else:
        ax2.scatter(data.index, data.H, color="navy")
        ax2.set_title("Hospitalizations")
    ax2.axvline(x = forecast1_start, color = 'darkgreen', linestyle ="--", label = 'begin forecast 1')
    ax2.axvline(x = forecast2_start, color = 'forestgreen', linestyle =":", label = 'begin forecast 2')
    ax2.axvline(x = forecast3_start, color = 'green', linestyle ="-.", label = 'begin forecast 3')
    ax2.axvline(x = forecast4_start, color = 'teal', label = 'begin forecast 4')
    ax2.legend(loc='lower right')
    
    if per100K:
        ax3.scatter(data.index, 100000*(data.D/N), color="blue")
        ax3.set_title("Cumulative Deaths per 100,000")
    else:
        ax3.scatter(data.index, data.D, color="blue")
        ax3.set_title("Cumulative Deaths")
    ax3.axvline(x = forecast1_start, color = 'darkgreen', linestyle ="--", label = 'begin forecast 1')
    ax3.axvline(x = forecast2_start, color = 'forestgreen', linestyle =":", label = 'begin forecast 2')
    ax3.axvline(x = forecast3_start, color = 'green', linestyle ="-.", label = 'begin forecast 3')
    ax3.axvline(x = forecast4_start, color = 'teal', label = 'begin forecast 4')
    ax3.legend(loc='lower right')

    return plt
    
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def reshape_multiindex_df_to_tensor(df: pd.DataFrame, indices=[ 'sample_id', 'timepoint_id']) -> dict[str, torch.tensor]:
    '''
    Reshape a multiindex dataframe to a dictionary of tensors.
    '''
    result = {}
    if not isinstance(df.index, pd.MultiIndex):
        df = df.set_index(indices)
    for col in df.columns:
        if '_sol' in col:
            values = df[col].values
            shape = [len(df.index.get_level_values(level).unique()) for level in df.index.names]
            tensor_values = torch.tensor(values).reshape(*shape)
            result[col] = tensor_values
    return result

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def process_and_save_quantiles(results: pd.DataFrame, q_filename: str, timepoints: list, train_data: pd.DataFrame, start_date: str, location_name: str):
    '''
    Calculate and format quantiles from results DataFrame
    '''
    results_dict = reshape_multiindex_df_to_tensor(results)
    quantiles_df = make_quantiles({'states': results_dict}, timepoints)
    quantiles_df.rename(columns={'number_(unknown)': 'number_days'}, inplace=True)
    quantiles_df.loc[quantiles_df['number_days'] < len(train_data), 'Forecast_Backcast'] = "Backcast"
    quantiles_df = quantiles_df[quantiles_df['number_days'] % 1 == 0]
    quantiles_df = quantiles_df[(quantiles_df['output'] == "Incident_Cases") | (quantiles_df['output'] == "Incident_Hosp") | (quantiles_df['output'] == "Incident_Deaths") | (quantiles_df['output'] == "D")]
    
    quantiles_df.to_csv(q_filename, index=False)
    q_ensemble_data = cdc_reformatcsv(filename=q_filename, 
                                      solution_string_mapping={"D": "cum death", "Incident_Cases": "inc case", "Incident_Hosp": "inc hosp", "Incident_Deaths": "inc death"}, 
                                      forecast_start_date=start_date,
                                      location=location_name,
                                      drop_column_names=["timepoint_id", "number_days", "inc_cum", "output", "Forecast_Backcast"])
    q_ensemble_data.to_csv(q_filename)
    return None

















