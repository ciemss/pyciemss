import numpy as np
import pytest

import pyciemss
from pyciemss.integration_utils.result_processing import convert_to_output_format
from pyciemss.visuals import plots, vega

# TODO: Consider testing some of these utils on raw schemas instead of with-data schemas


@pytest.fixture
def distributions():
    model_1_path = (
        "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/"
        "main/data/models/SEIRHD_NPI_Type1_petrinet.json"
    )
    start_time = 0.0
    end_time = 100.0
    logging_step_size = 1
    num_samples = 30
    sample = pyciemss.sample(
        model_1_path,
        end_time,
        logging_step_size,
        num_samples,
        start_time=start_time,
        solver_method="euler",
        solver_options={"step_size": 0.1},
    )["unprocessed_result"]

    for e in sample.values():
        if len(e.shape) > 1:
            num_timepoints = e.shape[1]

    return convert_to_output_format(
        sample,
        timepoints=np.linspace(start_time, end_time, num_timepoints),
        time_unit="notional",
    )[0]


def test_resize(distributions):
    w = 101
    h = 102

    schema1 = plots.trajectories(distributions)
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


def test_orient_legend(distributions):
    schema1 = plots.trajectories(distributions)

    new_orientation = "not-really-an-option"
    schema2 = plots.orient_legend(schema1, "color_legend", new_orientation)

    legend = vega.find_keyed(schema2["legends"], "name", "color_legend")
    assert legend["orient"] == new_orientation

    schema3 = plots.orient_legend(schema1, "color_legend", None)
    legend = vega.find_keyed(schema3["legends"], "name", "color_legend")
    assert "orient" not in legend


def test_pad(distributions):
    schema1 = plots.trajectories(distributions)
    schema2 = plots.pad(schema1, 5)
    schema3 = plots.pad(schema1, 20)
    schema4 = plots.pad(schema1, None)

    assert schema2["padding"] == 5
    assert schema3["padding"] == 20
    assert "padding" not in schema4


def test_title(distributions):
    schema1 = plots.trajectories(distributions)
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

    new_title = ["title", "with", "several", "parts"]
    schema = vega.set_title(schema1, new_title)
    assert schema["title"]["text"] == new_title


def test_rescale(distributions):
    pass


def test_replace_named_with(distributions):
    pass


def test_delete_named(distributions):
    schema1 = plots.trajectories(distributions)
    assert vega.find_keyed(schema1["signals"], "name", "clear") is not None

    schema_fragment = vega.delete_named(schema1["signals"], "clear")
    assert schema1["signals"] != schema_fragment, "Expected copy did not occur"

    with pytest.raises(ValueError):
        vega.find_keyed(schema_fragment, "name", "clear")


def test_find_keyed(distributions):
    schema1 = plots.trajectories(distributions)
    assert vega.find_keyed(schema1["signals"], "name", "clear") is not None
    with pytest.raises(ValueError):
        vega.find_keyed(schema1["signals"], "name", "NOT THERE")
