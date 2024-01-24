from pyciemss.visuals import plots, vega
from pathlib import Path
import pytest
import torch
import numpy as np
import json

from utils import convert_to_output_format, get_tspan


_data_root = Path(__file__).parent / "data"


def tensor_load(path):
    with open(path) as f:
        data = json.load(f)

    data = {k: torch.from_numpy(np.array(v)) for k, v in data.items()}

    return data


@pytest.fixture
def prior_dists():
    tspan = get_tspan(1, 50, 500).detach().numpy()
    return convert_to_output_format(
        tensor_load(_data_root / "prior_samples.json"),
        tspan,
        time_unit="notional",
    )


def test_resize(prior_dists):
    w = 101
    h = 102

    schema1 = plots.trajectories(prior_dists)
    schema2 = plots.resize(schema1, w=w, h=h)

    assert schema1 != schema2
    assert schema2["width"] == w
    assert schema2["height"] == h

    schema3 = plots.resize(schema1, w=w)
    assert schema3["width"] == w
    assert schema3["height"] == schema1["height"]

    schema4 = plots.resize(schema1, h=h)
    assert schema4["width"] == schema1["width"]
    assert schema4["height"] == h


def test_orient_legend(prior_dists):
    schema1 = plots.trajectories(prior_dists)

    new_orientation = "not-really-an-option"
    schema2 = plots.orient_legend(schema1, "color_legend", new_orientation)

    legend = vega.find_keyed(schema2["legends"], "name", "color_legend")
    assert legend["orient"] == new_orientation

    schema3 = plots.orient_legend(schema1, "color_legend", None)
    legend = vega.find_keyed(schema3["legends"], "name", "color_legend")
    assert "orient" not in legend


def test_pad(prior_dists):
    schema1 = plots.trajectories(prior_dists)
    schema2 = plots.pad(schema1, 5)
    schema3 = plots.pad(schema1, 20)
    schema4 = plots.pad(schema1, None)

    assert schema2["padding"] == 5
    assert schema3["padding"] == 20
    assert "padding" not in schema4


def test_title(prior_dists):
    schema1 = plots.trajectories(prior_dists)
    schema2 = plots.set_title(schema1, "Main Title")
    schema3 = plots.set_title(schema1, "XTitle", target="x")
    schema4 = plots.set_title(schema1, "YTitle", target="y")

    assert schema1 != schema2, "Expected copy did not occur"

    assert "title" not in schema1
    assert schema2["title"] == "Main Title"

    xaxis = vega.find_keyed(schema3["axes"], "name", "x_axis")
    yaxis = vega.find_keyed(schema3["axes"], "name", "y_axis")
    assert "title" not in schema3
    assert "title" not in yaxis
    assert xaxis["title"] == "XTitle"

    xaxis = vega.find_keyed(schema4["axes"], "name", "x_axis")
    yaxis = vega.find_keyed(schema4["axes"], "name", "y_axis")
    assert "title" not in schema4
    assert "title" not in xaxis
    assert yaxis["title"] == "YTitle"


def test_rescale(prior_dists):
    pass


def test_replace_named_with(prior_dists):
    pass


def test_delete_named(prior_dists):
    schema1 = plots.trajectories(prior_dists)
    assert vega.find_keyed(schema1["signals"], "name", "clear") is not None

    schema_fragment = vega.delete_named(schema1["signals"], "clear")
    assert schema1["signals"] != schema_fragment, "Expected copy did not occur"

    with pytest.raises(ValueError):
        vega.find_keyed(schema_fragment, "name", "clear")


def test_find_keyed(prior_dists):
    schema1 = plots.trajectories(prior_dists)
    assert vega.find_keyed(schema1["signals"], "name", "clear") is not None
    with pytest.raises(ValueError):
        vega.find_keyed(schema1["signals"], "name", "NOT THERE")
