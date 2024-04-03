import argparse
import difflib
import io
import json
import os
import re
import sys
from pathlib import Path

import IPython
import numpy as np
import pytest
from PIL import Image
from scipy.spatial.distance import jensenshannon
from xmldiff import main

from pyciemss.visuals import checks, plots

_schema_root = Path(__file__).parent.parent.parent / "pyciemss" / "visuals" / "schemas"

_reference_root = Path(__file__).parent / "reference_images"
_output_root = Path(__file__).parent / "output_images"


def save_result(data, name, ref_ext):
    """Save new reference files"""
    _output_root.mkdir(parents=True, exist_ok=True)

    mode = "w" if ref_ext == "svg" else "wb"
    with open(os.path.join(_output_root, f"{name}.{ref_ext}"), mode) as f:
        f.write(data)


def svg_matches(wrapped, ref_file):
    """Return the reference and svg created from schema

    wrapped -- IPython display SVG, contains the data property
    ref_file -- path to reference files
    returns -- Return the content and reference svg files
    """
    if not isinstance(wrapped, IPython.display.SVG):
        raise ValueError("Expected wrapped SVG")

    with open(ref_file) as f:
        reference = "".join(f.readlines())
        # replace what seems to be random numbers for gradient and cliip in svg
    reference = re.sub("gradient_?[0-9]*", "gradient_REPLACED", reference)
    reference = re.sub("clip[0-9]*", "clipREPLACED", reference)

    content = re.sub("gradient_?[0-9]*", "gradient_REPLACED", "".join(wrapped.data))
    content = re.sub("clip[0-9]*", "clipREPLACED", content)
    return content, reference


def background_white(orig_image):
    """Convert transparant background to white"""
    image = orig_image.convert("RGBA")
    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)
    new_image.convert("L")
    return new_image


def png_matches(image, ref_file, threshold):
    """Check how similiar the histograms
    of the reference and png created from schema are.

    image -- PNG result of plotting
    ref_file -- path to reference files
    returns -- boolean if jenson shannon value is under threshold
    """

    # open Image
    reference = Image.open(ref_file)
    content = Image.open(io.BytesIO(image))

    # background set to white
    new_reference = background_white(reference)
    new_content = background_white(content)
    # convert to list
    content_pixels = list(new_content.getdata())
    reference_pixels = list(new_reference.getdata())
    # [a-b for a, b in zip(content_pixels, reference_pixels) if a != b]
    content_hist, _ = np.histogram(content_pixels, bins=100)
    reference_hist, _ = np.histogram(reference_pixels, bins=100)
    # check if histograms are similiar enough
    return checks.JS(threshold, verbose=True)(
        content_hist, reference_hist
    ), jensenshannon(content_hist, reference_hist)


"""
Test that the schemas follow some common conventions.  There may be reason
to violate these conventions, but these are lint-like checks that may help
to avoid problems"""


def schemas(ref_ext=None):
    """
    Find all schema files.  If ref_ext is not None, figure out names for it
    """
    schemas = [*_schema_root.glob("*.vg.json")]
    schemas = [x for x in schemas if x.stem != "map_heatmap.vg"]

    assert len(schemas) > 0, "No schemas found"

    if ref_ext is not None:
        reference_names = {f"{schema.stem.split('.')[0]}": schema for schema in schemas}
        reference_files = {
            name: _reference_root / f"{name}.{ref_ext}"
            for name in reference_names.keys()
        }

        schemas = [
            (schema_file, reference_files[name], name)
            for name, schema_file in reference_names.items()
            if reference_files[name].exists()
        ]
        assert len(schemas) > 0, f"No schema with images type '{ref_ext}' found"

    return schemas


@pytest.mark.parametrize("schema_file", schemas())
def test_export_interactive(schema_file):
    with open(schema_file) as f:
        schema = json.load(f)

    image = plots.ipy_display(schema, format="interactive")

    # Interactive wraps for jupyter notebook, and displays it (does not return anything)
    assert image is None, f"Interactive failed for {schema_file}"


@pytest.mark.parametrize("schema_file, ref_file, name", schemas(ref_ext="png"))
def test_export_PNG(schema_file, ref_file, name):
    """
    Test all default schema files against the reference files for PNG files

    schema_file: default schema files saved within the visuals module
    ref_file: compare the created  png to this reference file
    name: stem name of reference file
    """
    with open(schema_file) as f:
        schema = json.load(f)

    image = plots.ipy_display(schema, format="PNG", dpi=72).data
    save_result(image, name, "png")

    # JS_score of 0.15 observed for some hardware choices.
    test_threshold = 0.2
    JS_boolean, JS_score = png_matches(image, ref_file, test_threshold)
    assert (
        JS_boolean
    ), f"{name}: PNG Histogram divergence: Shannon Jansen value {JS_score} > {test_threshold} "


@pytest.mark.parametrize("schema_file, ref_file, name", schemas(ref_ext="svg"))
def test_export_SVG(schema_file, ref_file, name):
    def svg_matches(result):
        with open(ref_file) as f:
            ref = "".join(f.readlines())

        diffa = main.diff_texts(result, ref)
        diffb = main.diff_texts(ref, result)

        for a, b in zip(diffa, diffb):
            if hasattr(a, "name") & hasattr(b, "name"):
                if a.name == b.name and a.name == "d":
                    ratio = difflib.SequenceMatcher(
                        a=a.value, b=b.value, autojunk=False
                    ).quick_ratio()
                    if ratio < 0.95:
                        return False
            else:
                # Assume its a name-issue and check it modulo numbers removed
                simple_a = re.sub(r"\d+", "", diffa[0].value).strip()
                simple_b = re.sub(r"\d+", "", diffb[0].value).strip()
                if simple_a != simple_b:
                    return False
        return True

    with open(schema_file) as f:
        schema = json.load(f)

    image = plots.ipy_display(schema, format="SVG").data
    save_result(image, name, "svg")
    assert svg_matches(image), f"{name}: SVG failed for {schema_file}"


@pytest.mark.parametrize("schema_file", schemas())
def test_color_legend_exists(schema_file):
    """If there is a color-focused scale, is there a legend for it?"""

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
                        find_scale_applications(value, target_properties, found=found)
                else:
                    find_scale_applications(value, target_properties, found=found)
        elif isinstance(schema, list):
            for entry in schema:
                find_scale_applications(entry, target_properties, found=found)

        return found

    schema_issues = []
    color_properties = ["fill", "stroke"]
    with open(schema_file) as f:
        schema = json.load(f)

    color_applications = find_scale_applications(schema["marks"], color_properties)

    color_legends = find_scale_applications(schema.get("legends", []), color_properties)

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


@pytest.mark.parametrize("schema_file", schemas())
def test_nested_mark_sources(schema_file):
    """If there is a group-mark, do the marks of that group point at the split-out-data?
    Group marks almost always follow from the split data,
    though it is (vega) syntatcitally valid to point at any datasource AND there might be
    a derived data-source as well.
    """

    with open(schema_file) as f:
        schema = json.load(f)

    group_marks = [m for m in schema["marks"] if m["type"] == "group"]
    if "trajectories.vg.json" == schema_file.name:
        assert (
            len(group_marks) == 5
        ), f"{schema_file.name} spot-check number of group marks incorrect"

    for group in group_marks:
        split_data = group.get("from", {}).get("facet", {}).get("name")

        if "trajectories.vg.json" == schema_file.name and group["name"] in [
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

            split_data = [split_data] + [d["name"] for d in group["data"]]
        else:
            split_data = [split_data]

        for mark in group["marks"]:
            assert (
                mark["from"]["data"] in split_data
            ), "Did not use split data (or derivation) in group mark"


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Utility to generate reference images")
    parser.add_argument(
        "schema",
        type=Path,
        help=f"Schema to load. Just the schema name, will look in {_schema_root} for the schema",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Execute but don't save files"
    )
    args = parser.parse_args()

    args.schema = _schema_root / args.schema
    png_output = _reference_root / f"{args.schema.stem.split('.')[0]}.png"
    svg_output = _reference_root / f"{args.schema.stem.split('.')[0]}.svg"

    if not args.schema.exists():
        print(f"Could not input file: {args.schema}")
        sys.exit(-1)

    with open(args.schema) as f:
        schema = json.load(f)

    print(f"Read from: {args.schema}")
    print(f"Target png: {png_output}")
    print(f"Target svg: {svg_output}")

    png = plots.ipy_display(schema, format="png")
    svg = plots.ipy_display(schema, format="svg")
    print("Rendering succeeded")

    if not args.dry_run:
        with open(png_output, "wb") as f:
            f.write(png.data)
        with open(svg_output, "w") as f:
            f.write(svg.data)
        print("References saved")
