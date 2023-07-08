from pyciemss.visuals import plots
from pathlib import Path
import unittest

from pyciemss.utils import get_tspan
from pyciemss.utils.interface_utils import convert_to_output_format

_data_root = Path(__file__).parent.parent / "data"


class TestUtils(unittest.TestCase):
    def setUp(self):
        tspan = get_tspan(1, 50, 500)
        self.dists = convert_to_output_format(
            plots.tensor_load(_data_root / "prior_samples.json"), tspan, nice=True
        )

    def test_resize(self):
        w = 101
        h = 102

        schema1 = plots.trajectories(self.dists)
        schema2 = plots.resize(schema1, w=w, h=h)

        self.assertFalse(schema1 == schema2)
        self.assertEqual(schema2["width"], w)
        self.assertEqual(schema2["height"], h)

        schema3 = plots.resize(schema1, w=w)
        self.assertEqual(schema3["width"], w)
        self.assertEqual(schema3["height"], schema1["height"])

        schema4 = plots.resize(schema1, h=h)
        self.assertEqual(schema4["width"], schema1["width"])
        self.assertEqual(schema4["height"], h)

    def test_orient_legend(self):
        schema1 = plots.trajectories(self.dists)

        new_orientation = "not-really-an-option"
        schema2 = plots.orient_legend(schema1, "color_legend", new_orientation)

        legend = plots.find_keyed(schema2["legends"], "name", "color_legend")
        self.assertEqual(legend["orient"], new_orientation)

        schema3 = plots.orient_legend(schema1, "color_legend", None)
        legend = plots.find_keyed(schema3["legends"], "name", "color_legend")
        self.assertFalse("orient" in legend)

    def test_pad(self):
        schema1 = plots.trajectories(self.dists)
        schema2 = plots.pad(schema1, 5)
        schema3 = plots.pad(schema1, 20)
        schema4 = plots.pad(schema1, None)

        self.assertEqual(schema2["padding"], 5)
        self.assertEqual(schema3["padding"], 20)
        self.assertFalse("padding" in schema4)

    def test_title(self):
        schema1 = plots.trajectories(self.dists)
        schema2 = plots.title(schema1, "Main Title")
        schema3 = plots.title(schema1, "XTitle", target="x")
        schema4 = plots.title(schema1, "YTitle", target="y")

        self.assertFalse(schema1 == schema2, "Expected copy did not occur")

        self.assertFalse("title" in schema1)
        self.assertEqual(schema2["title"], "Main Title")

        xaxis = plots.find_keyed(schema3["axes"], "name", "x_axis")
        yaxis = plots.find_keyed(schema3["axes"], "name", "y_axis")
        self.assertFalse("title" in schema3)
        self.assertFalse("title" in yaxis)
        self.assertEqual(xaxis["title"], "XTitle")

        xaxis = plots.find_keyed(schema4["axes"], "name", "x_axis")
        yaxis = plots.find_keyed(schema4["axes"], "name", "y_axis")
        self.assertFalse("title" in schema4)
        self.assertFalse("title" in xaxis)
        self.assertEqual(yaxis["title"], "YTitle")

    def test_rescale(self):
        pass

    def test_replace_named_with(self):
        pass

    def test_delete_named(self):
        schema1 = plots.trajectories(self.dists)
        self.assertIsNotNone(plots.find_keyed(schema1["signals"], "name", "clear"))

        schema_fragment = plots.delete_named(schema1["signals"], "clear")
        self.assertFalse(
            schema1["signals"] == schema_fragment, "Expected copy did not occur"
        )
        self.assertRaises(
            ValueError, plots.find_keyed, schema_fragment, "name", "clear"
        )

    def test_find_keyed(self):
        schema1 = plots.trajectories(self.dists)
        self.assertIsNotNone(plots.find_keyed(schema1["signals"], "name", "clear"))
        self.assertRaises(
            ValueError, plots.find_keyed, schema1["signals"], "name", "NOT THERE"
        )
