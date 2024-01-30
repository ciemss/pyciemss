from typing import Callable, Any
import pytest
import json
from pathlib import Path
import numpy as np

import IPython
from pyciemss.visuals import plots
from pyciemss.visuals import checks

from PIL import Image
from PIL import ImageChops

import os
import io
import re



_schema_root = (
    Path(__file__).parent.parent.parent / "pyciemss" / "visuals" / "schemas"
)

_reference_root = Path(__file__).parent / "reference_images"

create_reference_images = False

def save_png_svg(png_image, name, ref_ext):
    """Save new reference files"""
    if ref_ext == "png":
        with open(os.path.join(_reference_root, f"{name}.{ref_ext}"), "wb") as f:
            f.write(png_image.data)
    elif ref_ext == "svg":
        with open(os.path.join(_reference_root, f"{name}.{ref_ext}"), "w") as f:
            f.write(png_image.data)

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
    reference = re.sub('gradient_?[0-9]*', "gradient_REPLACED", reference)
    reference = re.sub('clip[0-9]*', "clipREPLACED", reference)
    
    content = re.sub('gradient_?[0-9]*', "gradient_REPLACED", "".join(wrapped.data))
    content = re.sub('clip[0-9]*', "clipREPLACED", content)
    return content, reference

def png_matches(schema, ref_file):
    """Check how similiar the histograms 
    of the reference and png created from schema are. 

    schema -- schema to create png file
    ref_file -- path to reference files
    returns -- boolean if jenson shannon value is under threshold
    """
    image = plots.ipy_display(schema, format="bytes", dpi=72) 
    reference = Image.open(ref_file)
    content = Image.open(io.BytesIO(image))
    content_pixels = list(content.convert('L').getdata())
    reference_pixels =  list(reference.convert('L').getdata())
    content_hist, edges = np.histogram(content_pixels,  bins=10)
    reference_hist, edges = np.histogram(reference_pixels,  bins=10)
    return checks.JS(0.1, verbose = True)(content_hist, reference_hist)


"""
Test that the schemas follow some common conventions.  There may be reason
to violate these conventions, but these are lint-like checks that may help
to avoid problems"""


def schemas(ref_ext=None):
    """  
    Find all schema files.  If ref_ext is not None, figure out names for
    """
    schemas = [*_schema_root.glob("*.vg.json")]
    assert len(schemas) > 0, "No schemas found"

    if ref_ext is not None:
        reference_names = [f"{schema.stem.split('.')[0]}"
            for schema in schemas
        ]
        reference_files = [
            _reference_root / f"{names}.{ref_ext}"
            for names in reference_names
        ]
        # if creating reference images, okay to keep schemas without pngs
        if not create_reference_images:
            reference_files = [f for f in reference_files if f.exists()]

        schemas = [*zip(schemas, reference_files, reference_names)]
        assert (
            len(schemas) > 0
        ), f"No schema with images type '{ref_ext}' found"

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

    If create_reference_image, recreate the reference files
    """
    with open(schema_file) as f:
        schema = json.load(f)

    image = plots.ipy_display(schema, format = "PNG", dpi=72)
    # create reference files if schema is new
    if create_reference_images:
        save_png_svg(image, name, "png")

    JS_boolean = png_matches(schema, ref_file)
    assert JS_boolean, "Histogram divergence: Shannon Jansen value is over 0.1"


@pytest.mark.parametrize("schema_file, ref_file, name", schemas(ref_ext="svg"))
def test_export_SVG(schema_file, ref_file, name):
    """  
    Test all default schema files against the reference files for SVG files

    schema_file: default schema files saved within the visuals module
    ref_file: compare the created svg to this reference file
    name: stem name of reference file

    If create_reference_image, recreate the reference files
    """
    with open(schema_file) as f:
        schema = json.load(f)

    image = plots.ipy_display(schema, format="SVG")
    # create reference files if schema is new
    if create_reference_images:
        save_png_svg(image, name, "svg")
    content, reference = svg_matches(image, ref_file)
    assert content == reference, f"SVG failed for {name}"



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
                        find_scale_applications(
                            value, target_properties, found=found
                        )
                else:
                    find_scale_applications(
                        value, target_properties, found=found
                    )
        elif isinstance(schema, list):
            for entry in schema:
                find_scale_applications(entry, target_properties, found=found)

        return found

    schema_issues = []
    color_properties = ["fill", "stroke"]
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
            len(group_marks) == 4
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