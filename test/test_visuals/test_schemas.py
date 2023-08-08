import unittest
import json
from pathlib import Path


"""
Test that the schemas follow some common conventions.  There may be reason
to violate these conventions, but these are lint-like checks that may help
to avoid problems"""


_schema_root = Path(__file__).parent.parent.parent / "src" / "pyciemss" / "visuals"


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


class TestTrajectory(unittest.TestCase):
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
        though it is (vega) syntatcitally valid to point at any datasource.
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
                    f"{schema_file.name} spot-check number of group marks",
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
                            f"{schema_file.name} spot-check facet found",
                        )

                    if split_data is None:
                        # Data not faceted
                        continue

                    for mark in group["marks"]:
                        self.assertEqual(
                            mark["from"]["data"],
                            split_data,
                            "Did not use split data in group mark",
                        )
            except Exception as e:
                schema_issues.append({"file": schema_file.name, "issue": str(e)})

        # NOTE: an empyt list is 'falsy' in python, so this fails when there are any entries
        self.assertFalse(schema_issues)
