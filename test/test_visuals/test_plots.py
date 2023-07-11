from pyciemss.visuals import plots
from pathlib import Path
import unittest
import pandas as pd
import xarray as xr

_data_root = Path(__file__).parent.parent / "data"


def by_key_value(targets, key, value):
    for entry in targets:
        if entry[key] == value:
            return entry


class TestHistograms(unittest.TestCase):
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

        raw_data = read_cube(_data_root / "ciemss_datacube.nc")
        self.s30 = raw_data.loc[
            (raw_data["time"] == 30) & (raw_data["state_names"] == "S")
        ]
        self.r30 = raw_data.loc[
            (raw_data["time"] == 30) & (raw_data["state_names"] == "R")
        ]
        self.i30 = raw_data.loc[
            (raw_data["time"] == 30) & (raw_data["state_names"] == "I")
        ]

    def test_histogram(self):
        hist, bins = plots.histogram_multi(s30=self.s30, return_bins=True)

        bins = bins.reset_index()

        self.assertTrue(all(bins["bin0"].value_counts() == 1), "Duplicated bins found")
        self.assertTrue(all(bins["bin1"].value_counts() == 1), "Duplicated bins found")

        hist_data = pd.DataFrame(by_key_value(hist["data"], "name", "binned")["values"])
        self.assertTrue(all(bins == hist_data))

        self.assertEqual(0, len(by_key_value(hist["data"], "name", "xref")["values"]))
        self.assertEqual(0, len(by_key_value(hist["data"], "name", "yref")["values"]))

    def test_histogram_empty_refs(self):
        xrefs = []
        yrefs = []
        hist, bins = plots.histogram_multi(
            s30=self.s30, xrefs=xrefs, yrefs=yrefs, return_bins=True
        )

        self.assertEqual(0, len(by_key_value(hist["data"], "name", "xref")["values"]))
        self.assertEqual(0, len(by_key_value(hist["data"], "name", "yref")["values"]))

    def test_histogram_refs(self):
        for num_refs in range(1, 20):
            xrefs = [*range(num_refs)]
            yrefs = [*range(num_refs)]
            hist, bins = plots.histogram_multi(
                s30=self.s30, xrefs=xrefs, yrefs=yrefs, return_bins=True
            )

            self.assertEqual(
                num_refs,
                len(by_key_value(hist["data"], "name", "xref")["values"]),
                "Nonzero xrefs not as expected",
            )
            self.assertEqual(
                num_refs,
                len(by_key_value(hist["data"], "name", "yref")["values"]),
                "Nonzero yrefs not as expected",
            )

            hist, bins = plots.histogram_multi(
                s30=self.s30, xrefs=xrefs, yrefs=[], return_bins=True
            )

            self.assertEqual(
                num_refs,
                len(by_key_value(hist["data"], "name", "xref")["values"]),
                "Nonzero xrefs not as expected when there are zero yrefs",
            )
            self.assertEqual(
                0,
                len(by_key_value(hist["data"], "name", "yref")["values"]),
                "Zero yrefs not as expected when there are nonzero xrefs",
            )

            hist, bins = plots.histogram_multi(
                s30=self.s30, xrefs=[], yrefs=yrefs, return_bins=True
            )

            self.assertEqual(
                0,
                len(by_key_value(hist["data"], "name", "xref")["values"]),
                "Zero xrefs not as expected when there are nonzero yrefs",
            )
            self.assertEqual(
                num_refs,
                len(by_key_value(hist["data"], "name", "yref")["values"]),
                "Nonzero yrefs not as expected when there are zero xrefs",
            )

    def test_histogram_multi(self):
        hist = plots.histogram_multi(s30=self.s30, r30=self.r30, i30=self.i30)
        data = pd.DataFrame(by_key_value(hist["data"], "name", "binned")["values"])
        self.assertEqual({"s30", "i30", "r30"}, set(data["label"].values))

        hist = plots.histogram_multi(s30=self.s30, r30=self.r30)
        data = pd.DataFrame(by_key_value(hist["data"], "name", "binned")["values"])
        self.assertEqual({"s30", "r30"}, set(data["label"].values))
