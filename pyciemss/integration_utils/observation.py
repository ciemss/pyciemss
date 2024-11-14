from typing import Dict, Tuple, Union

import pandas as pd
import torch

from pyciemss.observation import NoiseModel, NormalNoiseModel

_STR_TO_OBSERVATION = {"normal": NormalNoiseModel}


def load_data(
    path: Union[str, pd.DataFrame], data_mapping: Dict[str, str] = {}
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Load data from a CSV file, or directly from a DataFrame.

    - path: path to the CSV file, or DataFrame
    - data_mapping: A mapping from column names in the data file to state variable names in the model.
        - keys: str name of column in dataset
        - values: str name of state/observable in model
    - If not provided, we will assume that the column names in the data file match the state variable names.
    """

    def check_data(data_path: Union[str, pd.DataFrame]):
        # This function checks a dataset for formatting errors, and returns a DataFrame

        # Read the data
        if isinstance(data_path, str):
            # If data_path is a string, assume it's a file path and read as a csv
            data_df = pd.read_csv(data_path)
        elif isinstance(data_path, pd.DataFrame):
            # If data_path is a DataFrame, use it directly
            data_df = data_path
        else:
            # If data_path is neither a string nor a DataFrame, raise an error
            raise ValueError("data_path must be either a file path or a DataFrame.")

        # Check that the first column name is "Timestamp"
        if data_df.columns[0] != "Timestamp":
            raise ValueError(
                "The first column must be named 'Timestamp' and contain the time corresponding to each row of data."
            )

        # Check that there are no NaN values or empty entries
        if data_df.isna().any().any():
            raise ValueError("Dataset cannot contain NaN or empty entries.")

        # Check that there is no missing data in the form of None type or char values
        if not data_df.applymap(lambda x: isinstance(x, (int, float))).all().all():
            raise ValueError(
                "Dataset cannot contain None type or char values. All entries must be of type `int` or `float`."
            )

        return data_df

    def print_data_report(data_df):
        # Prints a short report about the data

        print(
            f"Data printout: This dataset contains {len(data_df) - 1} rows of data. "
            f"The first column, {data_df.columns[0]}, begins at {data_df.iloc[0, 0]} "
            f"and ends at {data_df.iloc[-1, 0]}. "
            f"The subsequent columns are named: "
            f"{', '.join(data_df.columns[1:])}"
        )

    df = check_data(path)
    print_data_report(df)

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


def identify_nans(data_df, return_labels=False):
    """
    Identifies and returns all types of NaNs in the DataFrame:
    1. Entire rows that are NaN.
    2. Entire columns that are NaN.
    3. Individual elements that are NaN.

    Parameters:
    data_df (pd.DataFrame): The input DataFrame.
    return_labels (bool): If True, return row and column names; otherwise, return indices.

    Returns:
    dict: A dictionary containing indices or labels for each type of NaN.
    """
    # Identify entire rows that are NaN
    nan_rows = data_df.index[data_df.isna().all(axis=1)].tolist()
    nan_row_idx = [data_df.index.get_loc(row) for row in nan_rows]

    # Identify entire columns that are NaN
    nan_columns = data_df.columns[data_df.isna().all(axis=0)].tolist()
    nan_columns_idx = [data_df.columns.get_loc(col) for col in nan_columns]

    # Identify individual elements that are NaN
    nan_elements = [(row, col) 
                        for row in data_df.index 
                        for col in data_df.columns 
                        if col not in nan_columns and
                         row not in nan_rows and 
                         pd.isna(data_df.loc[row, col])]

    nan_elements_idx = [(data_df.index.get_loc(row), data_df.columns.get_loc(col)) 
                            for row, col in nan_elements]
    return {
        "nan_rows": nan_rows if return_labels else nan_row_idx,
        "nan_columns": nan_columns if return_labels else nan_columns_idx,
        "nan_elements": nan_elements if return_labels else nan_elements_idx
    }
