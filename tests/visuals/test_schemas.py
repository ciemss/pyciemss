from typing import Callable, Any
import pytest
import json
from pathlib import Path

import IPython
from pyciemss.visuals import plots


"""
Test that the schemas follow some common conventions.  There may be reason
to violate these conventions, but these are lint-like checks that may help
to avoid problems"""


_schema_root = (
    Path(__file__).parent.parent.parent / "pyciemss" / "visuals" / "schemas"
)


def find_scale_applications(schema, target_properties, found=None):
    if found is None:
        found = []

    if isinstance(schema, dict):
        for key, value in schema.items():
            if key in target_properties:
                if isinstance(value, dict) and "scale" in value:
                    found.append(value["scale"])
                elif isinstance(value, str):
                    found.append(value)
                else:
                    find_scale_applications(
                        value, target_properties, found=found
                    )
            else:
                find_scale_applications(value, target_properties, found=found)
    elif isinstance(schema, list):
        for entry in schema:
            find_scale_applications(entry, target_properties, found=found)

    return found


@pytest.fixture
def schemas():
    schemas = [*_schema_root.glob("*.vg.json")]
    assert len(schemas) > 0, "No schemas found"
    return schemas


def format_checker(
    schemas, format: str, content_check: Callable[[str, Any], None]
):
    """
    Args:
        format (str): What file format to export to?
        content_check (Callable[[str, Any], None]):
            Test if the content matches a reference.
            Raise ValueError if it does not.

    Returns:
        _type_: _description_
    """
    schema_issues = []
    for schema_file in schemas:
        with open(schema_file) as f:
            schema = json.load(f)

        try:
            image = plots.ipy_display(schema, format=format)
            content_check(schema_file, image)
        except Exception as e:
            schema_issues.append({"file": schema_file.stem, "issue": str(e)})

    return schema_issues


# TODO: Turn into a parameterized over schemas
def test_export_interactive(schemas):
    def interactive_check(_, wrapped):
        if wrapped is not None:
            raise ValueError("Recieved a result when None was expected")

    schema_issues = format_checker(schemas, "interactive", interactive_check)
    assert len(schema_issues) == 0


# TODO: Turn into a parameterized over schemas
def test_export_PNG(schemas):
    saved_pngs = (Path(__file__).parent / "reference_images").glob("*.png")
    saved_images = {f.stem: f for f in saved_pngs}

    def png_check(schema_file, wrapped):
        if not isinstance(wrapped, IPython.display.Image):
            raise ValueError("Expected wrapped PNG")

        reference_file = saved_images.get(schema_file.stem, None)
        if reference_file is not None:
            with open(reference_file, "b") as f:
                reference = f.read()

            content = wrapped.data
            if content != reference:
                raise ValueError("PNG content does not match")

    schema_issues = format_checker(schemas, "PNG", png_check)
    assert len(schema_issues) == 0


# TODO: Turn into a parameterized over schemas
def test_export_SVG(schemas):
    saved_svgs = (Path(__file__).parent / "reference_images").glob("*.svg")
    saved_images = {f.stem: f for f in saved_svgs}

    def svg_check(schema_file, wrapped):
        if not isinstance(wrapped, IPython.display.SVG):
            raise ValueError("Expected wrapped SVG")

        reference_file = saved_images.get(schema_file.stem, None)
        if reference_file is not None:
            with open(reference_file) as f:
                reference = "".join(f.readlines(f))
            content = "".join(wrapped.data)
            if content != reference:
                raise ValueError("SVG content does not match")

        # TODO: Check contents of schema_file against stored images

    schema_issues = format_checker(schemas, "SVG", svg_check)
    assert len(schema_issues) == 0


# TODO: Turn into a parameterized over schemas
def test_color_legend_exists(schemas):
    """If there is a color-focused scale, is there a legend for it?"""

    schema_issues = []
    color_properties = ["fill", "stroke"]
    for schema_file in schemas:
        with open(schema_file) as f:
            schema = json.load(f)

        color_applications = find_scale_applications(
            schema["marks"], color_properties
        )

        color_legends = find_scale_applications(
            schema.get("legends", []), color_properties
        )

        if "trajectories.vg.json" == schema_file.name:
            assert (
                len(color_applications) == 6
            ), f"{schema_file.name} spot-check for color applications"
            assert (
                len(color_legends) == 1
            ), f"{schema_file.name} spot-check for color legends"

        color_applications = set(color_applications)
        color_legends = set(color_legends)

        if color_applications != color_legends:
            schema_issues.append(
                {
                    "file": schema_file.name,
                    "missing-legends": color_applications - color_legends,
                }
            )

    assert len(schema_issues) == 0


# TODO: Turn into a parameterized over schemas
def test_nested_mark_sources(schemas):
    """If there is a group-mark, do the marks of that group point at the split-out-data?
    Group marks almost always follow from the split data,
    though it is (vega) syntatcitally valid to point at any datasource AND there might be
    a derived data-source as well.
    """

    schema_issues = []
    for schema_file in schemas:
        with open(schema_file) as f:
            schema = json.load(f)

        group_marks = [m for m in schema["marks"] if m["type"] == "group"]
        if "trajectories.vg.json" == schema_file.name:
            assert (
                len(group_marks) == 4
            ), f"{schema_file.name} spot-check number of group marks incorrect"

        try:
            for group in group_marks:
                split_data = group.get("from", {}).get("facet", {}).get("name")

                if "trajectories.vg.json" == schema_file.name and group[
                    "name"
                ] in [
                    "_points",
                    "_traces",
                    "_distributions",
                ]:
                    assert (
                        split_data is not None
                    ), f"{schema_file.name} spot-check facet not found"

                if split_data is None:
                    # Data not faceted
                    continue

                if "data" in group:
                    # Can transform each facet...then the marks usually derive from here
                    datas = [d["source"] for d in group["data"]]
                    assert split_data in datas, "Facets not used in derived"

                    split_data = [split_data] + [
                        d["name"] for d in group["data"]
                    ]
                else:
                    split_data = [split_data]

                for mark in group["marks"]:
                    assert (
                        mark["from"]["data"] in split_data
                    ), "Did not use split data (or derivation) in group mark"

        except Exception as e:
            schema_issues.append({"file": schema_file.name, "issue": str(e)})

    assert len(schema_issues) == 0
