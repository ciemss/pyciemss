import json
from numbers import Number
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple, Union, overload

import numpy as np
import pandas as pd
from collections import defaultdict

from . import vega

_output_root = Path(__file__).parent / "data"


def calculate_mean_shapley(json_data):
    """
    Calculate the mean Shapley values for each column from the provided JSON data.
    
    Args:
        json_data : Json containing Shapley values and expected values.
    
    Returns:
        Json mean Shapley values and expected values for each column.
    """

    column_data = defaultdict(list)
    expected_value = {}
    
    for entry in json_data['values']:
        column = entry["Column"]
        column_data[column].append(entry["Shapley_Value"])
        expected_value[column] = entry["expected_value"]
    
    result = {"values": []}
    
    for column, values in column_data.items():
        mean_value = sum(values) / len(values)
        result["values"].append({
            "Column": column,
            "Mean Shapley Value": mean_value,
            "expected_value": expected_value[column]
        })
    
    return result

def process_explainer_output(explainer_output, return_mean_shapley=False):
    """
    Processes the output from a SHAP explainer and returns the SHAP values in a JSON-like format.
    Args:
        explainer_output (object): The output from a SHAP explainer, containing base values, SHAP values, feature names, and data.
        return_mean_shapley (bool, optional): If True, calculates and returns the mean SHAP values. Defaults to False.
    Returns:
        list: A list of dictionaries, each containing the SHAP value, column name, individual index, and expected value.
    """
    
    expected_value = explainer_output.base_values.mean(axis=0)
    are_base_values_constant = np.all(
        explainer_output.base_values == explainer_output.base_values[0]
    )
    print("Are all base_values the same?", are_base_values_constant)


    # Create DataFrames for SHAP values and data
    shap_values_df = pd.DataFrame(
        explainer_output.values, columns=explainer_output.feature_names
    )
    shap_data_df = pd.DataFrame(
        explainer_output.data, columns=explainer_output.feature_names
    )
    json_data = {"values": []}
    for i, row in shap_values_df.iterrows():
        for column in shap_values_df.columns:
            json_data["values"].append({
                "Column": column,
                "Shapley_Value": row[column],
                "individual": i + 1,
                "expected_value": expected_value
            })
    if return_mean_shapley:
        json_data = calculate_mean_shapley(json_data)
    return json_data['values']



def shapley_decision_plot(explainer_output) -> vega.VegaSchema:
    """
    Generates a Shapley decision plot using the provided explainer output.

    Parameters:
    explainer_output (Any): The output from a Shapley explainer which contains the data to be visualized.

    Returns:
    vega.VegaSchema: A Vega schema object representing the Shapley decision plot.
   

    """

    schema = vega.load_schema("shapley_decision_plot.vg.json")
    json_data = process_explainer_output(explainer_output)

    # load heatmap data
    schema["data"] = vega.replace_named_with(
    schema["data"], "table", ["values"], json_data
    )
    return schema


def shapley_bar_chart(explainer_output) -> vega.VegaSchema:
    """
    Generates a Shapley bar chart visualization using the provided explainer output.

    Parameters:
    explainer_output (dict): The output from a Shapley explainer, containing the Shapley values for different features.

    Returns:
    vega.VegaSchema: A Vega schema object representing the Shapley bar chart.
    """

    schema = vega.load_schema("shapley_bar_chart.vg.json")
    json_data = process_explainer_output(explainer_output, return_mean_shapley=True)

    # load heatmap data
    schema["data"] = vega.replace_named_with(
        schema["data"], "table", ["values"], json_data
    )
    return schema


def shapley_waterfall(explainer_output) -> vega.VegaSchema:
    """
    Generates a Shapley waterfall plot using the provided explainer output.

    Parameters:
    explainer_output (dict): The output from a Shapley explainer containing the Shapley values.

    Returns:
    vega.VegaSchema: A Vega schema object representing the Shapley waterfall plot.
    """

    schema = vega.load_schema("shapley_waterfall.vg.json")
    json_data = process_explainer_output(explainer_output, return_mean_shapley=True)

    # load heatmap data
    schema["data"] = vega.replace_named_with(
        schema["data"], "table", ["values"], json_data
    )
    return schema
