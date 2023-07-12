import unittest
import pandas as pd
import xarray as xr

from pathlib import Path

from pyciemss.visuals import plots
from pyciemss.utils import get_tspan
from pyciemss.utils.interface_utils import convert_to_output_format


_data_root = Path(__file__).parent.parent / "data"


def by_key_value(targets, key, value):
    for entry in targets:
        if entry[key] == value:
            return entry


class TestTrajectory(unittest.TestCase):
    def setUp(self):
        self.tspan = get_tspan(1, 50, 500).detach().numpy()
        self.nice_labels = {"Rabbits_sol": "Rabbits", "Wolves_sol": "Wolves"}

        self.dists = convert_to_output_format(
            plots.tensor_load(_data_root / "prior_samples.json"),
            self.tspan,
            time_unit="notional",
        )

        exemplars = self.dists[self.dists["sample_id"] == 0]
        wolves = exemplars.set_index("timepoint_notional")["Wolves_sol"].rename(
            "Wolves Example"
        )
        rabbits = exemplars.set_index("timepoint_notional")["Rabbits_sol"].rename(
            "Rabbits Example"
        )
        self.traces = pd.concat([wolves, rabbits], axis="columns")

        self.observed_trajectory = convert_to_output_format(
            plots.tensor_load(_data_root / "observed_trajectory.json"),
            self.tspan,
            time_unit="years",
        )

        self.observed_points = (
            self.observed_trajectory.rename(columns={"Rabbits_sol": "Rabbits Samples"})
            .drop(
                columns=[
                    "Wolves_sol",
                    "alpha_param",
                    "beta_param",
                    "delta_param",
                    "gamma_param",
                ]
            )
            .iloc[::10]
        )

    def test_base(self):
        schema = plots.trajectories(self.dists)

        df = pd.DataFrame(plots.find_named(schema["data"], "distributions")["values"])
        self.assertSetEqual(
            {"trajectory", "timepoint", "lower", "upper"}, set(df.columns)
        )

    def test_rename(self):
        schema = plots.trajectories(self.dists, relabel=self.nice_labels)

        df = pd.DataFrame(plots.find_named(schema["data"], "distributions")["values"])
        self.assertIn("Rabbits", df["trajectory"].unique())
        self.assertIn("Wolves", df["trajectory"].unique())
        self.assertNotIn("Rabbits_sol", df["trajectory"].unique())
        self.assertNotIn("Wolves_sol", df["trajectory"].unique())

    def test_subset(self):
        schema = plots.trajectories(self.dists, subset=".*_sol")
        df = pd.DataFrame(plots.find_named(schema["data"], "distributions")["values"])
        self.assertEqual(
            ["Rabbits_sol", "Wolves_sol"],
            sorted(df["trajectory"].unique()),
            "Subsetting by regex",
        )

        schema = plots.trajectories(self.dists, subset=["Rabbits_sol", "Wolves_sol"])
        df = pd.DataFrame(plots.find_named(schema["data"], "distributions")["values"])
        self.assertEqual(
            ["Rabbits_sol", "Wolves_sol"],
            sorted(df["trajectory"].unique()),
            "Subsetting by list",
        )

        schema = plots.trajectories(
            self.dists, relabel=self.nice_labels, subset=["Rabbits_sol", "Wolves_sol"]
        )
        df = pd.DataFrame(plots.find_named(schema["data"], "distributions")["values"])
        self.assertEqual(
            ["Rabbits", "Wolves"],
            sorted(df["trajectory"].unique()),
            "Rename after subsetting",
        )

    def test_points(self):
        schema = plots.trajectories(
            self.dists,
            subset=".*_sol",
            relabel=self.nice_labels,
            points=self.observed_points,
        )

        points = pd.DataFrame(plots.find_named(schema["data"], "points")["values"])
        print(points.columns)

        self.assertEqual(
            1, len(points["trajectory"].unique()), "Unexpected number of exemplars"
        )
        self.assertEqual(
            len(self.observed_points),
            len(points),
            "Unexpected number of exemplar points",
        )

    def test_traces(self):
        schema = plots.trajectories(
            self.dists,
            subset=".*_sol",
            relabel=self.nice_labels,
            traces=self.traces,
        )

        traces = pd.DataFrame(plots.find_named(schema["data"], "traces")["values"])
        plots.save_schema(schema, "_schema.json")

        self.assertEqual(
            sorted(self.traces.columns.unique()),
            sorted(traces["trajectory"].unique()),
            "Unexpected number of traces",
        )

        for exemplar in traces["trajectory"].unique():
            self.assertEqual(
                len(self.traces),
                len(traces[traces["trajectory"] == exemplar]),
                "Unexpected number of trace data points",
            )


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
