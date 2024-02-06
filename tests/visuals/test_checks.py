import numpy as np
import pandas as pd
import pytest

from pyciemss.visuals import checks, plots


def test_JS():
    d1 = [0, 0, 0, 0, 0, 5, 9, 5, 0]
    d2 = [5, 9, 5, 0, 0, 5, 9, 5, 0]
    d3 = [1, 2, 3, 4, 5, 5, 3, 2, 1]

    assert checks.JS(1)(d1, d1), "Identical distributions passes at 1"
    assert checks.JS(0)(d1, d1), "Identical distributions passes at 0"

    assert checks.JS(1)(d1, d2), "Disjoint passes at 1"
    assert not checks.JS(0.4)(d1, d2), "Disjoint should fail at .4"
    assert checks.JS(0.6)(d1, d3), "Overlap passes"
    assert checks.JS(0.6)(d2, d3), "Overlap passes"


def test_contains():
    _, bins = plots.histogram_multi(range=np.linspace(0, 20, num=100), return_bins=True)

    checker = checks.contains(3, 10)
    assert checker(bins), "In range"

    checker = checks.contains(-1, 10)
    assert not checker(bins), "Out lower"

    checker = checks.contains(3, 100)
    assert not checker(bins), "Out upper"

    checker = checks.contains(-10, 40)
    assert not checker(bins), "Out both"


def test_contains_pct():
    # TODO: This simple testing of percent-contains fails when there are
    #      not many things in the input.
    #      We should look at some better tests for small numbers of data points.
    _, bins = plots.histogram_multi(
        range=pd.Series(np.linspace(0, 20, num=100)), return_bins=True
    )

    checker = checks.contains(0, 20, 1)
    assert checker(bins), "Full range"

    checker = checks.contains(5, 15, 0.49)
    assert checker(bins), "Half middle"

    checker = checks.contains(15, 20, 0.24)
    assert checker(bins), ".25 upper"

    checker = checks.contains(0, 15, 0.74)
    assert checker(bins), ".75 lower"


@pytest.fixture
def normal0():
    return pd.Series(np.random.normal(0, size=200), name="normal-at-0")


@pytest.fixture
def normal2():
    return pd.Series(np.random.normal(2, size=200), name="normal-at-2")


def test_check_distribution_range(normal0):
    lower, upper = -2.0, 2.0
    result = checks.check_distribution_range(
        normal0,
        lower,
        upper,
        tests=[
            checks.contains(lower, upper),
            checks.contains(lower, upper, 0.99),
        ],
    )

    assert result.checks[0]
    assert not result.checks[1]
    assert "Fail" in result.schema["title"]["text"][1]
    assert "50%" in result.schema["title"]["text"][1]


def test_compare_distributions(normal0, normal2):
    result = checks.compare_distributions(normal0, normal2, tests=[checks.JS(0)])

    assert not result.status
    assert "0%" in result.schema["title"]["text"][1]

    result = checks.compare_distributions(normal0, normal2, tests=[checks.JS(1)])

    assert result.status
    assert "Pass" in result.schema["title"]["text"][1]
