from pyciemss.visuals import checks, plots
from pathlib import Path
import unittest
import numpy as np
import xarray as xr

_data_file = Path(__file__).parent.parent / "data" / "ciemss_datacube.nc"


class TestCheck(unittest.TestCase):
    def setUp(self):
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

        raw_data = read_cube(_data_file)
        self.s30 = raw_data.loc[
            (raw_data["time"] == 30) & (raw_data["state_names"] == "S")
        ]
        self.r30 = raw_data.loc[
            (raw_data["time"] == 30) & (raw_data["state_names"] == "R")
        ]
        self.i30 = raw_data.loc[
            (raw_data["time"] == 30) & (raw_data["state_names"] == "I")
        ]

    def test_JS(self):
        d1 = [0, 0, 0, 0, 0, 5, 9, 5, 0]
        d2 = [5, 9, 5, 0, 0, 5, 9, 5, 0]
        d3 = [1, 2, 3, 4, 5, 5, 3, 2, 1]

        self.assertTrue(checks.JS(1)(d1, d1), "Identical distributions passes at 1")
        self.assertTrue(checks.JS(0)(d1, d1), "Identical distributions passes at 0")
        self.assertTrue(checks.JS(1)(d1, d2), "Disjoint passes at 1")
        self.assertFalse(checks.JS(0.4)(d1, d2), "Disjoint fails at .9")
        self.assertTrue(checks.JS(0.6)(d1, d3), "Overlap passes")
        self.assertTrue(checks.JS(0.6)(d2, d3), "Overlap passes")

    def test_contains(self):
        _, bins = plots.histogram_multi(
            range=np.linspace(0, 20, num=100), return_bins=True
        )

        checker = checks.contains(3, 10)
        self.assertTrue(checker(bins), "In range")

        checker = checks.contains(-1, 10)
        self.assertFalse(checker(bins), "Out lower")

        checker = checks.contains(3, 100)
        self.assertFalse(checker(bins), "Out upper")

        checker = checks.contains(-10, 40)
        self.assertFalse(checker(bins), "Out both")

    def test_contains_pct(self):
        # TODO: This simple testing of percent-contains fails when there are
        #      not many things in the input.
        #      We should look at some better tests for small numbers of data points.
        _, bins = plots.histogram_multi(
            range=np.linspace(0, 20, num=100), return_bins=True, bins=20
        )

        checker = checks.contains(0, 20, 1)
        self.assertTrue(checker(bins), "Full range")

        checker = checks.contains(5, 15, 0.49)
        self.assertTrue(checker(bins), "Half middle")

        checker = checks.contains(15, 20, 0.24)
        self.assertTrue(checker(bins), ".25 upper")

        checker = checks.contains(0, 15, 0.74)
        self.assertTrue(checker(bins), ".75 lower")

    def test_check_distribution_range(self):
        lower, upper = 70_000, 94_000
        result = checks.check_distribution_range(
            self.s30,
            lower,
            upper,
            label="s30",
            tests=[checks.contains(lower, upper), checks.contains(lower, upper, 0.99)],
        )

        self.assertTrue(result.checks[0])
        self.assertFalse(result.checks[1])
        self.assertTrue("Fail" in result.schema["title"]["text"][1])
        self.assertTrue("50%" in result.schema["title"]["text"][1])

    def test_compare_distributions(self):
        result = checks.compare_distributions(self.s30, self.i30, tests=[checks.JS(0)])

        self.assertFalse(result.status)
        self.assertTrue("0%" in result.schema["title"]["text"][1])

        result = checks.compare_distributions(self.s30, self.s30, tests=[checks.JS(1)])

        self.assertTrue(result.status)
        self.assertTrue("Pass" in result.schema["title"]["text"][1])
