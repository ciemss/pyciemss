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
    column_data = defaultdict(list)
    expected_value = {}
    
    for entry in json_data:
        column = entry["Column"]
        column_data[column].append(entry["Shapley_Value"])
        expected_value[column] = entry["expected_value"]
    
    result = {"values": []}
    
    for column, values in column_data.items():
        mean_value = sum(values) / len(values)
        result["values"].append({
            "Column": column,
            "Mean Shapley Value": mean_value,
            "expected_value": expected_value[column] * len(values)
        })
    
    return result

def process_explainer_output(explainer_output, return_mean_shapley=False):
    # Extract Expected Value (Base Values)
    # This assumes base_values is a vector
    expected_value = explainer_output.base_values.mean(axis=0)
    are_base_values_constant = np.all(
        explainer_output.base_values == explainer_output.base_values[0]
    )
    print("Are all base_values the same?", are_base_values_constant)

    # Check if expected_value is scalar or vector
    if expected_value.ndim == 0:  # This is for scalar value
        expected_value = [expected_value]

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
    return json_data



def shapley_decision_plot(explainer_output) -> vega.VegaSchema:
    """
    mesh -- (Optional) pd.DataFrame with columns
        lon_start, lon_end, lat_start, lat_end, count for each grid
    """

    schema = vega.load_schema("shapley_decision_plot.vg.json")
    json_data = process_explainer_output(explainer_output)

    # load heatmap data
    schema["data"] = vega.replace_named_with(
        schema["data"], "table", ["values"], json_data
    )
    return schema


import xgboost
import shap
import pandas as pd
import numpy as np
# Train XGBoost model
X, y = shap.datasets.adult(n_points=100)
model = xgboost.XGBClassifier().fit(X, y)

# Compute SHAP values
explainer = shap.Explainer(model, X)
shap_values = explainer(X)


new_schema = shapley_decision_plot(shap_values)
