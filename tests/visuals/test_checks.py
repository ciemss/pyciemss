from pyciemss.visuals import checks, plots
from pathlib import Path
import pytest
import numpy as np
import xarray as xr

_data_file = Path(__file__).parent / "data" / "ciemss_datacube.nc"


def test_JS():
    d1 = [0, 0, 0, 0, 0, 5, 9, 5, 0]
    d2 = [5, 9, 5, 0, 0, 5, 9, 5, 0]
    d3 = [1, 2, 3, 4, 5, 5, 3, 2, 1]

    assert checks.JS(1)(d1, d1), "Identical distributions passes at 1"
    assert checks.JS(0)(d1, d1), "Identical distributions passes at 0"

    assert checks.JS(1)(d1, d2), "Disjoint passes at 1"
    assert checks.JS(0.4)(d1, d2) == False, "Disjoint should fail at .4"
    assert checks.JS(0.6)(d1, d3), "Overlap passes"
    assert checks.JS(0.6)(d2, d3), "Overlap passes"


def test_contains():
    _, bins = plots.histogram_multi(
        range=np.linspace(0, 20, num=100), return_bins=True
    )

    checker = checks.contains(3, 10)
    assert checker(bins), "In range"

    checker = checks.contains(-1, 10)
    assert checker(bins) == False, "Out lower"

    checker = checks.contains(3, 100)
    assert checker(bins) == False, "Out upper"

    checker = checks.contains(-10, 40)
    assert checker(bins) == False, "Out both"


def test_contains_pct():
    # TODO: This simple testing of percent-contains fails when there are
    #      not many things in the input.
    #      We should look at some better tests for small numbers of data points.
    _, bins = plots.histogram_multi(
        range=np.linspace(0, 20, num=100), return_bins=True, bins=20
    )

    checker = checks.contains(0, 20, 1)
    assert checker(bins), "Full range"

    checker = checks.contains(5, 15, 0.49)
    assert checker(bins), "Half middle"

    checker = checks.contains(15, 20, 0.24)
    assert checker(bins), ".25 upper"

    checker = checks.contains(0, 15, 0.74)
    assert checker(bins), ".75 lower"


def read_cube(file):
    ds = xr.open_mfdataset([file])
    real_data = ds.to_dataframe().reset_index()
    real_data.rename(
        columns={
            "timesteps": "time",
            "experimental conditions": "conditions",
            "attributes": "state_names",
            "__xarray_dataarray_variable__": "state_values",
        },
        inplace=True,
    )
    return real_data


@pytest.fixture
def datacube_s30():
    raw_data = read_cube(_data_file)
    return raw_data.loc[
        (raw_data["time"] == 30) & (raw_data["state_names"] == "S")
    ]


@pytest.fixture
def datacube_r30():
    raw_data = read_cube(_data_file)
    return raw_data.loc[
        (raw_data["time"] == 30) & (raw_data["state_names"] == "R")
    ]


@pytest.fixture
def datacube_i30():
    raw_data = read_cube(_data_file)
    return raw_data.loc[
        (raw_data["time"] == 30) & (raw_data["state_names"] == "I")
    ]


def test_check_distribution_range(datacube_s30):
    lower, upper = 70_000, 94_000
    result = checks.check_distribution_range(
        datacube_s30,
        lower,
        upper,
        label="s30",
        tests=[
            checks.contains(lower, upper),
            checks.contains(lower, upper, 0.99),
        ],
    )

    assert result.checks[0]
    assert result.checks[1] == False
    assert "Fail" in result.schema["title"]["text"][1]
    assert "50%" in result.schema["title"]["text"][1]


def test_compare_distributions(datacube_s30, datacube_i30):
    result = checks.compare_distributions(
        datacube_s30, datacube_i30, tests=[checks.JS(0)]
    )

    assert result.status == False
    assert "0%" in result.schema["title"]["text"][1]

    result = checks.compare_distributions(
        datacube_s30, datacube_s30, tests=[checks.JS(1)]
    )

    assert result.status
    assert "Pass" in result.schema["title"]["text"][1]
