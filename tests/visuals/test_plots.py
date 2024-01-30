import pytest
import pandas as pd
import xarray as xr
import numpy as np
import networkx as nx
import json
import torch
import random
from pathlib import Path
from itertools import chain

from pyciemss.visuals import plots, vega
from test_schemas import png_matches, svg_matches, save_png_svg
from utils import convert_to_output_format, get_tspan
import os

_data_root = Path(__file__).parent / "data"


_modified_schema_root = (
    Path(__file__).parent.parent.parent /  "tests" / "visuals" / "modified_schemas"
)

_reference_root = Path(__file__).parent / "reference_images"


# True if want to save reference files for modified schemas
create_reference_images = False
def save_schema(schema, name):
    """Save the modified schema to test again reference files"""
    _modified_schema_root.mkdir(parents=True, exist_ok=True)
    plots.save_schema(schema, os.path.join(_modified_schema_root, f"{name}.vg.json"))

def check_modified_images(schema, name, ref_ext):
    """check if the created images mathc the reference images 
    for either png or svg files

    schema-- modified schema 
    name -- name of reference file 
    ref_ext -- reference file extension 
    """
    image = plots.ipy_display(schema, format=ref_ext, dpi=72)
    # create reference files if schema is new
    if create_reference_images:
        save_png_svg(image, name, ref_ext)

    reference_file = _reference_root / f"{name}.{ref_ext}"
    if ref_ext == "png": 
        JS_boolean = png_matches(schema, reference_file)
        assert JS_boolean, "Histogram divergence: Shannon Jansen value is over 0.1"

    if ref_ext == "svg":
        content, reference = svg_matches(image, reference_file)
        assert content == reference, f"SVG failed for {name}"


def check_modified_schema_png(schema, name):
        
        """for each schema tested, save the schema, and check that
        the resulting svg and png files match the reference png and svg files

        schema-- modified schema 
        name -- name of reference file 
        """
        save_schema(schema, name)
        check_modified_images(schema, name, "png")
        # check_modified_images(schema, name, "svg")

def check_mismatch_mod_default(schema, default_name):
        
        """for each schema tested, save the schema, and check that
        the resulting svg and png files match the reference png and svg files

        schema-- modified schema 
        name -- name of reference file 
        """
        reference_file = _reference_root / f"{default_name}.png"
        JS_boolean = png_matches(schema, reference_file)
        assert not JS_boolean, "Histogram similiarity to default image: Shannon Jansen value is under 0.1"

def tensor_load(path):
    with open(path) as f:
        data = json.load(f)

    data = {k: torch.from_numpy(np.array(v)) for k, v in data.items()}

    return data


def by_key_value(targets, key, value):
    for entry in targets:
        if entry[key] == value:
            return entry


class TestTrajectory:
    @pytest.fixture
    def trajectory(self):
        self.tspan = get_tspan(1, 50, 500).detach().numpy()
        self.nice_labels = {"Rabbits_sol": "Rabbits", "Wolves_sol": "Wolves"}

        self.dists = convert_to_output_format(
            tensor_load(_data_root / "prior_samples.json"),
            self.tspan,
            time_unit="notional",
        )

        exemplars = self.dists[self.dists["sample_id"] == 0]
        wolves = exemplars.set_index("timepoint_notional")[
            "Wolves_sol"
        ].rename("Wolves Example")
        rabbits = exemplars.set_index("timepoint_notional")[
            "Rabbits_sol"
        ].rename("Rabbits Example")
        self.traces = pd.concat([wolves, rabbits], axis="columns")

        self.observed_trajectory = convert_to_output_format(
            tensor_load(_data_root / "observed_trajectory.json"),
            self.tspan,
            time_unit="years",
        )

        self.observed_points = (
            self.observed_trajectory.rename(
                columns={"Rabbits_sol": "Rabbits Samples"}
            )
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

    def test_base(self, trajectory):
        schema = plots.trajectories(self.dists)

        df = pd.DataFrame(
            vega.find_named(schema["data"], "distributions")["values"]
        )
        # save schemas so can check if created svg and png files match
        check_modified_schema_png(schema, "test_base")

        assert {"trajectory", "timepoint", "lower", "upper"} == set(df.columns)

    def test_rename(self, trajectory):
        schema = plots.trajectories(self.dists, relabel=self.nice_labels)
        # save schemas so can check if created svg and png files match
        check_modified_schema_png(schema, "test_rename")
        check_mismatch_mod_default(schema, "test_base")

        df = pd.DataFrame(
            vega.find_named(schema["data"], "distributions")["values"]
        )

        assert "Rabbits" in df["trajectory"].unique()
        assert "Wolves" in df["trajectory"].unique()
        assert "Rabbits_sol" not in df["trajectory"].unique()
        assert "Wolves_sol" not in df["trajectory"].unique()

    def test_keep(self, trajectory):
        schema = plots.trajectories(self.dists, keep=".*_sol")
        # save schemas so can check if created svg and png files match
        check_modified_schema_png(schema, "test_keep_sol")
        check_mismatch_mod_default(schema, "test_base")
        df = pd.DataFrame(
            vega.find_named(schema["data"], "distributions")["values"]
        )

        assert ["Rabbits_sol", "Wolves_sol"] == sorted(
            df["trajectory"].unique()
        ), "Keeping by regex"

        schema = plots.trajectories(
            self.dists, keep=["Rabbits_sol", "Wolves_sol"]
        )
        # save schemas so can check if created svg and png files match
        check_modified_schema_png(schema, "test_keep_named")
        check_mismatch_mod_default(schema, "test_base")

        df = pd.DataFrame(
            vega.find_named(schema["data"], "distributions")["values"]
        )
        assert ["Rabbits_sol", "Wolves_sol"] == sorted(
            df["trajectory"].unique()
        ), "Keeping by list"

        schema = plots.trajectories(
            self.dists,
            relabel=self.nice_labels,
            keep=["Rabbits_sol", "Wolves_sol"],
        )
        # save schemas so can check if created svg and png files match
        check_modified_schema_png(schema, "test_keep_nice_labels")
        check_mismatch_mod_default(schema, "test_base")

        df = pd.DataFrame(
            vega.find_named(schema["data"], "distributions")["values"]
        )
        assert ["Rabbits", "Wolves"] == sorted(
            df["trajectory"].unique()
        ), "Rename after Keeping"

    def test_keep_drop(self, trajectory):
        assert (
            "Rabbits_sol" in self.dists.columns
        ), "Exepected trajectory not found in pre-test"

        should_drop = [p for p in self.dists.columns if "_param" in p]
        assert (
            len(should_drop) > 0
        ), "Exepected trajectory not found in pre-test"

        schema = plots.trajectories(self.dists, keep=".*_.*", drop=".*_param")

        # save schemas so can check if created svg and png files match
        check_modified_schema_png(schema, "test_keep_drop")
        check_mismatch_mod_default(schema, "test_base")
        df = pd.DataFrame(
            vega.find_named(schema["data"], "distributions")["values"]
        )

        assert (
            "Rabbits_sol" in df["trajectory"].unique()
        ), "Exepected trajectory not retained from list"

        kept = [t for t in df["trajectory"].unique() if "_param" in t]
        assert len(kept) == 0, "Kept unexpexted columns in keep & drop case"

    def test_drop(self, trajectory):
        assert (
            "Rabbits_sol" in self.dists.columns
        ), "Exepected trajectory not found in pre-test"

        print(self.dists.columns)
        should_drop = [p for p in self.dists.columns if "_param" in p]
        assert (
            len(should_drop) > 0
        ), "Exepected trajectory not found in pre-test"

        schema = plots.trajectories(self.dists, drop=should_drop)
        check_modified_schema_png(schema, "test_should_drop")
        check_mismatch_mod_default(schema, "test_base")

        df = pd.DataFrame(
            vega.find_named(schema["data"], "distributions")["values"]
        )

        assert (
            "Rabbits_sol" in df["trajectory"].unique()
        ), "Exepected trajectory not retained from list"
        assert (
            "Wolves_sol" in df["trajectory"].unique()
        ), "Exepected trajectory not retained from list"

        for t in should_drop:
            assert (
                t not in df["trajectory"]
            ), "Trajectory still present after drop from list"

        try:
            schema = plots.trajectories(self.dists, drop="THIS IS NOT HERE")
        except Exception:
            assert False, "Error dropping non-existent trajectory"

        schema = plots.trajectories(self.dists, drop="gam.*")
        check_modified_schema_png(schema, "test_drop_gam")
        check_mismatch_mod_default(schema, "test_base")
        df = pd.DataFrame(
            vega.find_named(schema["data"], "distributions")["values"]
        )
        assert (
            "Rabbits_sol" in df["trajectory"].unique()
        ), "Exepected trajectory not retained from pattern"
        assert (
            "Wolves_sol" in df["trajectory"].unique()
        ), "Exepected trajectory not retained from pattern"
        assert (
            "gamma_param" not in df["trajectory"].unique()
        ), "Trajectory still present after drop from pattern"

    def test_points(self, trajectory):
        schema = plots.trajectories(
            self.dists,
            keep=".*_sol",
            relabel=self.nice_labels,
            points=self.observed_points,
        )
        check_modified_schema_png(schema, "test_traj_points")
        check_mismatch_mod_default(schema, "test_base")

        points = pd.DataFrame(
            vega.find_named(schema["data"], "points")["values"]
        )

        assert (
            len(points["trajectory"].unique()) == 1
        ), "Unexpected number of exemplars"
        assert len(self.observed_points) == len(
            points
        ), "Unexpected number of exemplar points"

    def test_traces(self, trajectory):
        schema = plots.trajectories(
            self.dists,
            keep=".*_sol",
            relabel=self.nice_labels,
            traces=self.traces,
        )
        check_modified_schema_png(schema, "test_traj_traces")
        check_mismatch_mod_default(schema, "test_base")

        traces = pd.DataFrame(
            vega.find_named(schema["data"], "traces")["values"]
        )

        assert sorted(self.traces.columns.unique()) == sorted(
            traces["trajectory"].unique()
        ), "Unexpected traces"

        for exemplar in traces["trajectory"].unique():
            assert len(self.traces) == len(
                traces[traces["trajectory"] == exemplar]
            ), "Unexpected number of trace data points"


class TestHistograms:
    @pytest.fixture
    def load_data(self):
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

    def test_histogram(self, load_data):
        hist, bins = plots.histogram_multi(s30=self.s30, return_bins=True)

        bins = bins.reset_index()

        assert all(bins["bin0"].value_counts() == 1), "Duplicated bins found"
        assert all(bins["bin1"].value_counts() == 1), "Duplicated bins found"

        hist_data = pd.DataFrame(
            by_key_value(hist["data"], "name", "binned")["values"]
        )
        assert all(bins == hist_data)

        assert len(by_key_value(hist["data"], "name", "xref")["values"]) == 0
        assert len(by_key_value(hist["data"], "name", "yref")["values"]) == 0

    def test_histogram_empty_refs(self, load_data):
        xrefs = []
        yrefs = []
        hist, bins = plots.histogram_multi(
            s30=self.s30, xrefs=xrefs, yrefs=yrefs, return_bins=True
        )

        assert len(by_key_value(hist["data"], "name", "xref")["values"]) == 0
        assert len(by_key_value(hist["data"], "name", "yref")["values"]) == 0

    @pytest.mark.parametrize("num_refs", range(1, 20))
    def test_histogram_refs(self, num_refs, load_data):
        xrefs = [*range(num_refs)]
        yrefs = [*range(num_refs)]
        hist, bins = plots.histogram_multi(
            s30=self.s30, xrefs=xrefs, yrefs=yrefs, return_bins=True
        )

        assert num_refs == len(
            by_key_value(hist["data"], "name", "xref")["values"]
        ), "Nonzero xrefs not as expected"
        assert num_refs == len(
            by_key_value(hist["data"], "name", "yref")["values"]
        ), "Nonzero yrefs not as expected"

        hist, bins = plots.histogram_multi(
            s30=self.s30, xrefs=xrefs, yrefs=[], return_bins=True
        )

        assert num_refs == len(
            by_key_value(hist["data"], "name", "xref")["values"]
        ), "Nonzero xrefs not as expected when there are zero yrefs"
        assert (
            len(by_key_value(hist["data"], "name", "yref")["values"]) == 0
        ), "Zero yrefs not as expected when there are nonzero xrefs"

        hist, bins = plots.histogram_multi(
            s30=self.s30, xrefs=[], yrefs=yrefs, return_bins=True
        )

        assert (
            len(by_key_value(hist["data"], "name", "xref")["values"]) == 0
        ), "Zero xrefs not as expected when there are nonzero yrefs"
        assert num_refs == len(
            by_key_value(hist["data"], "name", "yref")["values"]
        ), "Nonzero yrefs not as expected when there are zero xrefs"

    def test_histogram_multi(self, load_data):
        hist = plots.histogram_multi(s30=self.s30, r30=self.r30, i30=self.i30)
        data = pd.DataFrame(
            by_key_value(hist["data"], "name", "binned")["values"]
        )
        assert set(data["label"].values) == {"s30", "i30", "r30"}

        hist = plots.histogram_multi(s30=self.s30, r30=self.r30)
        data = pd.DataFrame(
            by_key_value(hist["data"], "name", "binned")["values"]
        )
        assert set(data["label"].values) == {"s30", "r30"}


class TestHeatmapScatter:
    def test_implicit_heatmap(self):
        np.random.seed(2)
        df = pd.DataFrame(
            3 * np.random.random((100, 2)), columns=["test4", "test5"]
        )
        schema = plots.heatmap_scatter(df, max_x_bins=4, max_y_bins=4)
        check_modified_schema_png(schema, "test_heatmap")

        points = vega.find_named(schema["data"], "points")["values"]
        assert all(
            pd.DataFrame(points) == df
        ), "Unexpected points values found"

    def test_explicit_heatmap(self):
        def create_fake_data():
            np.random.seed(2)
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
        check_modified_schema_png(schema, "test_heatmap_explicit")
        check_mismatch_mod_default(schema, "test_heatmap")

        points = vega.find_named(schema["data"], "points")["values"]
        assert all(
            pd.DataFrame(points) == scatter_data
        ), "Unexpected points values found"

        mesh = pd.DataFrame(vega.find_named(schema["data"], "mesh")["values"])
        assert mesh.size == 500, "Unexpected mesh representation size."
        assert all(
            mesh["__count"].isin(mesh_data[2].ravel())
        ), "Unexpected count found"


class TestGraph:
    @staticmethod
    @pytest.fixture
    def test_graph():
        random.seed(1)
        def rand_attributions():
            possible = "ABCD"
            return random.sample(possible, random.randint(1, len(possible)))

        def rand_label():
            possible = "TUVWXYZ"
            return random.randint(1, 10)
            return random.sample(possible, 1)[0]

        g = nx.generators.barabasi_albert_graph(5, 3)
        node_properties = {
            n: {"attribution": rand_attributions(), "label": rand_label()}
            for n in g.nodes()
        }

        edge_attributions = {
            e: {"attribution": rand_attributions()} for e in g.edges()
        }

        nx.set_node_attributes(g, node_properties)
        nx.set_edge_attributes(g, edge_attributions)
        return g

    def test_multigraph(self, test_graph):
        uncollapsed = plots.attributed_graph(test_graph)
        nodes = vega.find_named(uncollapsed["data"], "node-data")["values"]
        edges = vega.find_named(uncollapsed["data"], "link-data")["values"]
        assert len(test_graph.nodes) == len(nodes), "Nodes issue in conversion"
        assert len(test_graph.edges) == len(edges), "Edges issue in conversion"

        all_attributions = set(
            chain(*nx.get_node_attributes(test_graph, "attribution").values())
        )
        nx.set_node_attributes(
            test_graph, {0: {"attribution": all_attributions}}
        )
        collapsed = plots.attributed_graph(test_graph, collapse_all=True)
        nodes = vega.find_named(collapsed["data"], "node-data")["values"]
        edges = vega.find_named(collapsed["data"], "link-data")["values"]
        assert len(test_graph.nodes) == len(
            nodes
        ), "Nodes issue in conversion (collapse-case)"
        assert len(test_graph.edges) == len(
            edges
        ), "Edges issue in conversion (collapse-case)"
        assert [["*all*"]] == [
            n["attribution"] for n in nodes if n["label"] == 0
        ], "All tag not found as expected"

    def test_springgraph(self, test_graph):
        schema = plots.spring_force_graph(test_graph, node_labels="label")
        # random outlay from barabasi_albert_graph
        # check_modified_schema_png(schema, "tests_springgraph")
        nodes = vega.find_named(schema["data"], "node-data")["values"]
        edges = vega.find_named(schema["data"], "link-data")["values"]
        assert len(test_graph.nodes) == len(nodes), "Nodes issue in conversion"
        assert len(test_graph.edges) == len(edges), "Edges issue in conversion"

    def test_provided_layout(self, test_graph):
        pos = nx.fruchterman_reingold_layout(test_graph)
        schema = plots.spring_force_graph(
            test_graph, node_labels="label", layout=pos
        )
        # random layout so can't check sv
        #check_modified_schema_png(schema, "tests_springgraph_layout")

        nodes = vega.find_named(schema["data"], "node-data")["values"]
        edges = vega.find_named(schema["data"], "link-data")["values"]
        assert len(test_graph.nodes) == len(nodes), "Nodes issue in conversion"
        assert len(test_graph.edges) == len(edges), "Edges issue in conversion"

        for id, (x, y) in pos.items():
            n = [n for n in nodes if n["label"] == id][0]
            assert n["inputX"] == x, f"Layout lost for {id}"
            assert n["inputY"] == y, f"Layout lost for {id}"
