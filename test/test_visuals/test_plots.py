import unittest
import pandas as pd
import xarray as xr
import numpy as np
import networkx as nx
import json
import torch
import random
import os

from pathlib import Path
from itertools import chain

from pyciemss.visuals import plots, vega, trajectories
from pyciemss.utils import get_tspan
from pyciemss.utils.interface_utils import convert_to_output_format


_data_root = Path(__file__).parent.parent / "data"

save_schema = (
        Path(__file__).parent.parent / "test_visuals" / "modified_schemas" 
    )

save_png = (
    Path(__file__).parent.parent /  "test_visuals" / "reference_images" 
)

# True if want to save png and svg files
create_modified_schemas = False

def save_schema_png_svg(schema, name):
    plots.save_schema(schema, os.path.join(save_schema, name + ".vg.json"))
    png_image = plots.ipy_display(schema)

    with open(os.path.join(save_png, name + ".png"), "wb") as f:
        f.write(png_image.data)

    svg_image = plots.ipy_display(schema, format = "SVG")

    with open(os.path.join(save_png, name + ".svg"), "w") as f:
        f.write(svg_image.data)

def tensor_load(path):
    with open(path) as f:
        data = json.load(f)

    data = {k: torch.from_numpy(np.array(v)) for k, v in data.items()}

    return data


def by_key_value(targets, key, value):
    for entry in targets:
        if entry[key] == value:
            return entry


class TestTrajectory(unittest.TestCase):
    def setUp(self):
        self.tspan = get_tspan(1, 50, 500).detach().numpy()
        self.nice_labels = {"Rabbits_sol": "Rabbits", "Wolves_sol": "Wolves"}

        self.dists = convert_to_output_format(
            tensor_load(_data_root / "prior_samples.json"),
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
            tensor_load(_data_root / "observed_trajectory.json"),
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
        # save schemas so can check if created svg and png files match
        if create_modified_schemas:
            save_schema_png_svg(schema, "test_base")

        df = pd.DataFrame(vega.find_named(schema["data"], "distributions")["values"])
        self.assertSetEqual(
            {"trajectory", "timepoint", "lower", "upper"}, set(df.columns)
        )

    def test_rename(self):
        schema = plots.trajectories(self.dists, relabel=self.nice_labels)
        # save schemas so can check if created svg and png files match
        if create_modified_schemas:
            save_schema_png_svg(schema, "test_rename")

        df = pd.DataFrame(vega.find_named(schema["data"], "distributions")["values"])
        self.assertIn("Rabbits", df["trajectory"].unique())
        self.assertIn("Wolves", df["trajectory"].unique())
        self.assertNotIn("Rabbits_sol", df["trajectory"].unique())
        self.assertNotIn("Wolves_sol", df["trajectory"].unique())

    def test_keep(self):
        schema = plots.trajectories(self.dists, keep=".*_sol")
        # save schemas so can check if created svg and png files match
        if create_modified_schemas:
            save_schema_png_svg(schema, "test_keep")

        df = pd.DataFrame(vega.find_named(schema["data"], "distributions")["values"])
        self.assertEqual(
            ["Rabbits_sol", "Wolves_sol"],
            sorted(df["trajectory"].unique()),
            "Keeping by regex",
        )

        df = pd.DataFrame(vega.find_named(schema["data"], "distributions")["values"])
        self.assertEqual(
            ["Rabbits_sol", "Wolves_sol"],
            sorted(df["trajectory"].unique()),
            "Keeping by list",
        )

        schema = plots.trajectories(
            self.dists, relabel=self.nice_labels, keep=["Rabbits_sol", "Wolves_sol"]
        )

        # save schemas so can check if created svg and png files match
        if create_modified_schemas:
            save_schema_png_svg(schema, "test_keep_nice_labels")

        df = pd.DataFrame(vega.find_named(schema["data"], "distributions")["values"])
        self.assertEqual(
            ["Rabbits", "Wolves"],
            sorted(df["trajectory"].unique()),
            "Rename after Keeping",
        )

    def test_keep_drop(self):
        self.assertIn(
            "Rabbits_sol",
            self.dists.columns,
            "Exepected trajectory not found in pre-test",
        )

        should_drop = [p for p in self.dists.columns if "_param" in p]
        self.assertGreater(
            len(should_drop), 0, "Exepected trajectory not found in pre-test"
        )

        schema = plots.trajectories(self.dists, keep=".*_.*", drop=".*_param")

        # save schemas so can check if created svg and png files match
        if create_modified_schemas:
            save_schema_png_svg(schema, "test_keep_drop")
            
        df = pd.DataFrame(vega.find_named(schema["data"], "distributions")["values"])

        self.assertIn(
            "Rabbits_sol",
            df["trajectory"].unique(),
            "Exepected trajectory not retained from list",
        )

        kept = [t for t in df["trajectory"].unique() if "_param" in t]
        self.assertEqual(0, len(kept), "Kept unexpexted columns in keep & drop case")

    def test_drop(self):
        self.assertIn(
            "Rabbits_sol",
            self.dists.columns,
            "Exepected trajectory not found in pre-test",
        )

        should_drop = [p for p in self.dists.columns if "_param" in p]
        self.assertGreater(
            len(should_drop), 0, "Exepected trajectory not found in pre-test"
        )

        schema = plots.trajectories(self.dists, drop=should_drop)

        # save schemas so can check if created svg and png files match
        if create_modified_schemas:
            save_schema_png_svg(schema, "test_drop")

        df = pd.DataFrame(vega.find_named(schema["data"], "distributions")["values"])

        self.assertIn(
            "Rabbits_sol",
            df["trajectory"].unique(),
            "Exepected trajectory not retained from list",
        )
        self.assertIn(
            "Wolves_sol",
            df["trajectory"].unique(),
            "Exepected trajectory not retained from list",
        )

        for t in should_drop:
            self.assertNotIn(
                t,
                df["trajectory"].unique(),
                "Trajectory still present after drop from list",
            )

        try:
            schema = plots.trajectories(self.dists, drop="THIS IS NOT HERE")
        except Exception:
            self.fail("Error dropping non-existent trajectory")

        schema = plots.trajectories(self.dists, drop="gam.*")


        # save schemas so can check if created svg and png files match
        if create_modified_schemas:
            save_schema_png_svg(schema, "test_drop_gam")

        df = pd.DataFrame(vega.find_named(schema["data"], "distributions")["values"])
        self.assertIn(
            "Rabbits_sol",
            df["trajectory"].unique(),
            "Exepected trajectory not retained from pattern",
        )
        self.assertIn(
            "Wolves_sol",
            df["trajectory"].unique(),
            "Exepected trajectory not retained from pattern",
        )
        self.assertNotIn(
            "gamma_param",
            df["trajectory"].unique(),
            "Trajectory still present after drop from pattern",
        )

    def test_points(self):
        schema = plots.trajectories(
            self.dists,
            keep=".*_sol",
            relabel=self.nice_labels,
            points=self.observed_points,
        )

        # save schemas so can check if created svg and png files match
        if create_modified_schemas:
            save_schema_png_svg(schema, "test_points")

        points = pd.DataFrame(vega.find_named(schema["data"], "points")["values"])

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
            keep=".*_sol",
            relabel=self.nice_labels,
            traces=self.traces,
        )
        # save schemas so can check if created svg and png files match
        if create_modified_schemas:
            save_schema_png_svg(schema, "test_traces")

        traces = pd.DataFrame(vega.find_named(schema["data"], "traces")["values"])
        
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

    def test_mean_traces(self):
        traces = trajectories.select_traces(self.dists,
                                    select_by = "mean",
                                    keep=".*_sol", 
                                    relabel=self.nice_labels)
        # check no repeat values for time points
        self.assertEqual(len(np.unique(traces["timepoint_id"])), len(traces["timepoint_id"]))
        
        # check keeping only trajectories for kept columns
        traces_df = trajectories._nice_df(self.dists)
        traces_df = trajectories._keep_drop_rename(traces_df, ".*_sol", None, self.nice_labels)
        # check if equal without the timepoint columns (last column)
        traces_columns = list(traces.columns)
        traces_columns.remove('timepoint_id')
        self.assertCountEqual(traces_columns, list(traces_df.columns))
        
    def test_mean_traces_error(self):
         with self.assertRaises(ValueError):
            traces = trajectories.select_traces(self.dists,
                                    select_by = "mean2",
                                    keep=".*_sol", 
                                    relabel=self.nice_labels)



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

        # save schemas so can check if created svg and png files match
        if create_modified_schemas:
            save_schema_png_svg(hist, "test_histogram")

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

        # save schemas so can check if created svg and png files match
        if create_modified_schemas:
            save_schema_png_svg(hist, "test_histogram_empty_refs")

        # save schemas so can check if created svg and png files match
        if create_modified_schemas:
            save_schema_png_svg(hist, "test_histogram")


        self.assertEqual(0, len(by_key_value(hist["data"], "name", "xref")["values"]))
        self.assertEqual(0, len(by_key_value(hist["data"], "name", "yref")["values"]))

    def test_histogram_refs(self):
        for num_refs in range(1, 20):
            xrefs = [*range(num_refs)]
            yrefs = [*range(num_refs)]
            hist, bins = plots.histogram_multi(
                s30=self.s30, xrefs=xrefs, yrefs=yrefs, return_bins=True
            )

            # save schemas so can check if created svg and png files match
            if create_modified_schemas:
                save_schema_png_svg(hist, "test_histogram_refs")

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

            # save schemas so can check if created svg and png files match
            if create_modified_schemas:
                save_schema_png_svg(hist, "test_histogram_refs_xrefs")

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

            # save schemas so can check if created svg and png files match
            if create_modified_schemas:
                save_schema_png_svg(hist, "test_histogram_refs_yrefs")


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

        # save schemas so can check if created svg and png files match
        if create_modified_schemas:
            save_schema_png_svg(hist, "test_histogram_multi_sri")

        data = pd.DataFrame(by_key_value(hist["data"], "name", "binned")["values"])
        self.assertEqual({"s30", "i30", "r30"}, set(data["label"].values))

        hist = plots.histogram_multi(s30=self.s30, r30=self.r30)

        # save schemas so can check if created svg and png files match
        if create_modified_schemas:
            save_schema_png_svg(hist, "test_histogram_multi_sr")
            
        data = pd.DataFrame(by_key_value(hist["data"], "name", "binned")["values"])
        self.assertEqual({"s30", "r30"}, set(data["label"].values))


class TestHeatmapScatter(unittest.TestCase):
    def test_implicit_heatmap(self):
        df = pd.DataFrame(3 * np.random.random((100, 2)), columns=["test4", "test5"])
        schema = plots.heatmap_scatter(df, max_x_bins=4, max_y_bins=4)
        # save schemas so can check if created svg and png files match
        if create_modified_schemas:
            save_schema_png_svg(schema, "test_implicit_heatmap")
            
        points = vega.find_named(schema["data"], "points")["values"]
        self.assertTrue(
            all(pd.DataFrame(points) == df), "Unexpected points values found"
        )

    def test_explicit_heatmap(self):
        def create_fake_data():
            nx, ny = (10, 10)
            x = np.linspace(0, 10, nx)
            y, a = np.linspace(0, 10, ny, retstep=True)

            # create mesh data
            xv, yv = np.meshgrid(x, y)
            zz = xv**2 + yv**2

            # create scatter plot
            df = pd.DataFrame(
                10 * np.random.random((100, 2)), columns=["alpha", "gamma"]
            )
            return (xv, yv, zz), df

        mesh_data, scatter_data = create_fake_data()
        schema = plots.heatmap_scatter(scatter_data, mesh_data)
        # save schemas so can check if created svg and png files match
        if create_modified_schemas:
            save_schema_png_svg(schema, "test_explicit_heatmap")
            

        points = vega.find_named(schema["data"], "points")["values"]
        self.assertTrue(
            all(pd.DataFrame(points) == scatter_data), "Unexpected points values found"
        )

        mesh = pd.DataFrame(vega.find_named(schema["data"], "mesh")["values"])
        self.assertEqual(500, mesh.size, "Unexpected mesh representation size.")
        self.assertTrue(
            all(mesh["__count"].isin(mesh_data[2].ravel())), "Unexpected count found"
        )


class TestGraph(unittest.TestCase):
    def setUp(self):
        def rand_attributions():
            possible = "ABCD"
            return random.sample(possible, random.randint(1, len(possible)))

        def rand_label():
            possible = "TUVWXYZ"
            return random.randint(1, 10)
            return random.sample(possible, 1)[0]

        self.g = nx.generators.barabasi_albert_graph(5, 3)
        node_properties = {
            n: {"attribution": rand_attributions(), "label": rand_label()}
            for n in self.g.nodes()
        }

        edge_attributions = {
            e: {"attribution": rand_attributions()} for e in self.g.edges()
        }

        nx.set_node_attributes(self.g, node_properties)
        nx.set_edge_attributes(self.g, edge_attributions)

    def test_multigraph(self):
        uncollapsed = plots.attributed_graph(self.g)
        # save schemas so can check if created svg and png files match
        if create_modified_schemas:
            save_schema_png_svg(uncollapsed, "test_multigraph")

        nodes = vega.find_named(uncollapsed["data"], "node-data")["values"]
        edges = vega.find_named(uncollapsed["data"], "link-data")["values"]
        self.assertEqual(len(self.g.nodes), len(nodes), "Nodes issue in conversion")
        self.assertEqual(len(self.g.edges), len(edges), "Edges issue in conversion")

        all_attributions = set(
            chain(*nx.get_node_attributes(self.g, "attribution").values())
        )
        nx.set_node_attributes(self.g, {0: {"attribution": all_attributions}})
        collapsed = plots.attributed_graph(self.g, collapse_all=True)

        # save schemas so can check if created svg and png files match
        if create_modified_schemas:
            save_schema_png_svg(collapsed, "test_multigraph_collapsed")

        nodes = vega.find_named(collapsed["data"], "node-data")["values"]
        edges = vega.find_named(collapsed["data"], "link-data")["values"]
        self.assertEqual(
            len(self.g.nodes), len(nodes), "Nodes issue in conversion (collapse-case)"
        )
        self.assertEqual(
            len(self.g.edges), len(edges), "Edges issue in conversion (collapse-case)"
        )
        self.assertEqual(
            [["*all*"]],
            [n["attribution"] for n in nodes if n["label"] == 0],
            "All tag not found as expected",
        )

    def test_springgraph(self):
        schema = plots.spring_force_graph(self.g, node_labels="label")
        # save schemas so can check if created svg and png files match
        if create_modified_schemas:
            save_schema_png_svg(schema, "test_springgraph")

        nodes = vega.find_named(schema["data"], "node-data")["values"]
        edges = vega.find_named(schema["data"], "link-data")["values"]
        self.assertEqual(len(self.g.nodes), len(nodes), "Nodes issue in conversion")
        self.assertEqual(len(self.g.edges), len(edges), "Edges issue in conversion")

    def test_provided_layout(self):
        pos = nx.fruchterman_reingold_layout(self.g)
        schema = plots.spring_force_graph(self.g, node_labels="label", layout=pos)
        # save schemas so can check if created svg and png files match
        if create_modified_schemas:
            save_schema_png_svg(schema, "test_provided_layout")

        nodes = vega.find_named(schema["data"], "node-data")["values"]
        edges = vega.find_named(schema["data"], "link-data")["values"]
        self.assertEqual(len(self.g.nodes), len(nodes), "Nodes issue in conversion")
        self.assertEqual(len(self.g.edges), len(edges), "Edges issue in conversion")

        for id, (x, y) in pos.items():
            n = [n for n in nodes if n["label"] == id][0]
            self.assertEqual(n["inputX"], x, f"Layout lost for {id}")
            self.assertEqual(n["inputY"], y, f"Layout lost for {id}")
