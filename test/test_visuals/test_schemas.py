from typing import Callable, Any
import unittest
import json
from pathlib import Path
import re

import IPython
from pyciemss.visuals import plots

from PIL import Image
from PIL import ImageChops
import io


"""
Test that the schemas follow some common conventions.  There may be reason
to violate these conventions, but these are lint-like checks that may help
to avoid problems"""


_schema_root = (
    Path(__file__).parent.parent.parent / "src" / "pyciemss" / "visuals" / "schemas"
)

_modified_schema_root = (
        Path(__file__).parent.parent / "test_visuals" / "modified_schemas" 
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
                    find_scale_applications(value, target_properties, found=found)
            else:
                find_scale_applications(value, target_properties, found=found)
    elif isinstance(schema, list):
        for entry in schema:
            find_scale_applications(entry, target_properties, found=found)

    return found


class TestExport(unittest.TestCase):
    def setUp(self):
        self.schemas = [*_schema_root.glob("*.vg.json")]
        self.schemas_modified = [*_modified_schema_root.glob("*.vg.json")]
        self.all_schemas = self.schemas + self.schemas_modified
        
        self.assertGreater(len(self.schemas), 0, "No schemas found")

    def format_checker(self, format: str, content_check: Callable[[str, Any], None]):
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
        for schema_file in self.schemas_modified:
            with open(schema_file) as f:
                schema = json.load(f)
                print(schema_file.name)

            try:
                image = plots.ipy_display(schema, format=format, dpi=200) 
                content_check(schema_file, image)
            except Exception as e:
                schema_issues.append({"file": schema_file.stem, "issue": str(e)})
        
        return schema_issues

    def test_export_interactive(self):
        def interactive_check(_, wrapped):
            if wrapped is not None:
                raise ValueError("Recieved a result when None was expected")

        schema_issues = self.format_checker("interactive", interactive_check)
        self.assertFalse(schema_issues)

    def test_export_PNG(self):
        saved_pngs = (Path(__file__).parent / "reference_images").glob("*.png")
        saved_images = {f.stem: f for f in saved_pngs}

        def png_check(schema_file, wrapped):
            '''
            schema_file: json file with vega schema
            wrapped: IPython.display.PNG data
            '''
            # if not isinstance(wrapped, IPython.display.Image):
            #     raise ValueError("Expected bytes object")

            reference_file = saved_images.get(schema_file.name.split('.')[0], None)
            if reference_file is not None:
                reference = Image.open(reference_file)

                content = Image.open(io.BytesIO(wrapped))
                diff = ImageChops.difference(content.convert("RGB"), reference.convert("RGB"))
                diff_hist = diff.histogram()
                if len([x for x in diff_hist if x != 0]) > 3:
                    raise ValueError("PNG content does not match")

        schema_issues = self.format_checker("bytes", png_check)

        self.assertFalse(schema_issues)

    def test_export_SVG(self):
        saved_svgs = (Path(__file__).parent / "reference_images").glob("*.svg")
        saved_images = {f.stem: f for f in saved_svgs}

        def svg_check(schema_file, wrapped):
            '''
            schema_file: json file with vega schema
            wrapped: IPython.display.SVG data
            '''
            if not isinstance(wrapped, IPython.display.SVG):
                raise ValueError("Expected wrapped SVG")

            reference_file = saved_images.get(schema_file.stem.split(".")[0], None)
            if reference_file is not None:
                with open(reference_file, 'r') as f:
                    reference = f.read()
                    # replace what seems to be random numbers for gradient and cliip in svg
                    reference = re.sub('gradient_?[0-9]*', "gradient_REPLACED", reference)
                    reference = re.sub('clip[0-9]*', "clipREPLACED", reference)
                    
                content = re.sub('gradient_?[0-9]*', "gradient_REPLACED", wrapped.data)
                content = re.sub('clip[0-9]*', "clipREPLACED", content)
                if content != reference:
                    raise ValueError("SVG content does not match")

            # TODO: Check contents of schema_file against stored images

        schema_issues = self.format_checker("SVG", svg_check)
        self.assertFalse(schema_issues)


class TestSchemaContents(unittest.TestCase):
    def setUp(self):
        self.schemas = [*_schema_root.glob("*.vg.json")]
        self.assertGreater(len(self.schemas), 0, "No schemas found")

    def test_color_legend_exists(self):
        """If there is a color-focused scale, is there a legend for it?"""

        schema_issues = []
        color_properties = ["fill", "stroke"]
        for schema_file in self.schemas:
            with open(schema_file) as f:
                schema = json.load(f)

            color_applications = find_scale_applications(
                schema["marks"], color_properties
            )

            color_legends = find_scale_applications(
                schema.get("legends", []), color_properties
            )

            if "trajectories.vg.json" == schema_file.name:
                self.assertEqual(
                    len(color_applications),
                    6,
                    f"{schema_file.name} spot-check for color applications",
                )
                self.assertEqual(
                    len(color_legends),
                    1,
                    f"{schema_file.name} spot-check for color legends",
                )

            color_applications = set(color_applications)
            color_legends = set(color_legends)

            if color_applications != color_legends:
                schema_issues.append(
                    {
                        "file": schema_file.name,
                        "missing-legends": color_applications - color_legends,
                    }
                )

        # NOTE: an empyt list is 'falsy' in python, so this fails when there are any entries
        self.assertFalse(schema_issues)

    def test_nested_mark_sources(self):
        """If there is a group-mark, do the marks of that group point at the split-out-data?
        Group marks almost always follow from the split data,
        though it is (vega) syntatcitally valid to point at any datasource AND there might be
        a derived data-source as well.
        """

        schema_issues = []
        for schema_file in self.schemas:
            with open(schema_file) as f:
                schema = json.load(f)

            group_marks = [m for m in schema["marks"] if m["type"] == "group"]
            if "trajectories.vg.json" == schema_file.name:
                self.assertEqual(
                    len(group_marks),
                    4,
                    f"{schema_file.name} spot-check number of group marks incorrect",
                )

            try:
                for group in group_marks:
                    split_data = group.get("from", {}).get("facet", {}).get("name")

                    if "trajectories.vg.json" == schema_file.name and group["name"] in [
                        "_points",
                        "_traces",
                        "_distributions",
                    ]:
                        self.assertIsNotNone(
                            split_data,
                            f"{schema_file.name} spot-check facet not found",
                        )

                    if split_data is None:
                        # Data not faceted
                        continue

                    if "data" in group:
                        # Can transform each facet...then the marks usually derive from here
                        datas = [d["source"] for d in group["data"]]
                        self.assertIn(split_data, datas, "Facets not used in derived")
                        split_data = [split_data] + [d["name"] for d in group["data"]]
                    else:
                        split_data = [split_data]

                    for mark in group["marks"]:
                        self.assertIn(
                            mark["from"]["data"],
                            split_data,
                            "Did not use split data (or derivation) in group mark",
                        )
            except Exception as e:
                schema_issues.append({"file": schema_file.name, "issue": str(e)})

        # NOTE: an empyt list is 'falsy' in python, so this fails when there are any entries
        self.assertFalse(schema_issues)
