from typing import Dict, Tuple

import pandas as pd
import torch

from pyciemss.observation import NoiseModel, NormalNoiseModel

_STR_TO_OBSERVATION = {"normal": NormalNoiseModel}


def load_data(
    path: str, data_mapping: Dict[str, str] = {}
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Load data from a CSV file.

    - path: path to the CSV file
    - data_mapping: A mapping from column names in the data file to state variable names in the model.
        - keys: str name of column in dataset
        - values: str name of state/observable in model
    - If not provided, we will assume that the column names in the data file match the state variable names.
    """

    def check_data(data_path: str):
        # This function checks a dataset formatting errors, and returns a dataframe
        data_df = pd.read_csv(path)
        print(data_df.head())
        
        # Check that the first column name is "Timestamp"
        assert data_df.columns[0] == "Timestamp", "The column of timepoints must be first, and named 'Timestamp'."
        
        # Check that there are no NaN values or empty entries
        assert not data_df.isna().any().any(), "Dataset cannot contain NaN or empty entries"
        
        # Check that there is no missing data in the form of None type or char values
        assert data_df.applymap(lambda x: isinstance(x, (int, float))).all().all(), "Dataset cannot contain None type or char values. All entries must be of type `int` or `float`." 
        
        return data_df
    
    df = check_data(path)

    data_timepoints = torch.tensor(df["Timestamp"].values, dtype=torch.float32)
    data = {}

    for col in df.columns:
        if col == "Timestamp":
            continue

        if col in data_mapping:
            data[data_mapping[col]] = torch.tensor(df[col].values, dtype=torch.float32)
        else:
            data[col] = torch.tensor(df[col].values, dtype=torch.float32)
        # TODO: address missing data

    return data_timepoints, data


def compile_noise_model(model_str: str, **model_kwargs) -> NoiseModel:
    if model_str not in _STR_TO_OBSERVATION.keys():
        raise NotImplementedError(
            f"""Noise model {model_str} not implemented. /n
            Please select from one of the following: {_STR_TO_OBSERVATION.keys()}"""
        )

    return _STR_TO_OBSERVATION[model_str](**model_kwargs)
