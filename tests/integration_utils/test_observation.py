import pytest
import pandas as pd
from pyciemss.integration_utils.observation import identify_nans  # Assuming the function is in observation.py

@pytest.fixture
def nan_data():
    data = {
        'A': [1, 2, None, 4],
        'B': [None, None, None, None],
        'C': [None, None, None, None],
        'D': [1, 2, None, None]
    }
    return pd.DataFrame(data)

def test_identify_nans_indices(nan_data):
    nan_info = identify_nans(nan_data, return_labels=False)

    # Expected results
    expected_nan_rows = [2]
    expected_nan_columns = [1, 2]
    expected_nan_elements = [(3, 3)]

    assert nan_info["nan_rows"] == expected_nan_rows
    assert nan_info["nan_columns"] == expected_nan_columns
    assert nan_info["nan_elements"] == expected_nan_elements

def test_identify_nans_labels(nan_data):
    nan_info = identify_nans(nan_data, return_labels=True)

    # Expected results
    expected_nan_rows = [2]
    expected_nan_columns = ['B', 'C']
    expected_nan_elements = [(3, 'D')]

    assert nan_info["nan_rows"] == expected_nan_rows
    assert nan_info["nan_columns"] == expected_nan_columns
    assert nan_info["nan_elements"] == expected_nan_elements

# Run the tests
if __name__ == "__main__":
    pytest.main()