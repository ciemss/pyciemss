import unittest
import pandas as pd
import xarray as xr
import numpy as np
import json
import torch

from pathlib import Path

from pyciemss.visuals import plots, vega, graphs
import networkx as nx
import random

_data_root = Path(__file__).parent.parent / "data"

def rand_attributions():
    possible = "ABCD"
    return random.sample(possible, random.randint(1, len(possible)))

def rand_label():
    return random.randint(1, 10)

def rand_group():
    possible = "TUVWXYZ"
    return random.sample(possible, 1)[0]

class TestGraphs(unittest.TestCase):

    def test_graph_attribute(self):
        g = nx.generators.barabasi_albert_graph(5, 3)
        node_properties = {n: {"attribution": rand_attributions(), 
                            "label": rand_label()}
                            for n in g.nodes()}

        edge_attributions = {e: {"attribution": rand_attributions()} for e in g.edges()}

        nx.set_node_attributes(g, node_properties)
        nx.set_edge_attributes(g, edge_attributions)

        gjson = nx.json_graph.node_link_data(g)

        new_schema = plots.attributed_graph(g)

        self.assertEqual(
            len(vega.find_named(new_schema['data'], "node-data")['values']),  len(gjson["nodes"])
        )
        self.assertEqual(
            len(vega.find_named(new_schema['data'], "link-data")['values']),  len(gjson["links"])
        )



    def test_node_missing_attribute(self):
        with self.assertRaises(ValueError):
                g = nx.generators.barabasi_albert_graph(5, 3)
                node_properties = {n: {"label": rand_label()}
                                    for n in g.nodes()}

                edge_attributions = {e: {"attribution": rand_attributions()} for e in g.edges()}

                nx.set_node_attributes(g, node_properties)
                nx.set_edge_attributes(g, edge_attributions)
                schema = graphs.attributed_graph(g)

    def test_edge_missing_attribute(self):
        with self.assertRaises(ValueError):
                g = nx.generators.barabasi_albert_graph(5, 3)
                node_properties = {n: {"label": rand_label()}
                                    for n in g.nodes()}
                nx.set_node_attributes(g, node_properties)
                schema = graphs.attributed_graph(g)


class TestSpringGraph(unittest.TestCase):
    def fake_network(self):
        g = nx.generators.barabasi_albert_graph(5, 3)

        node_properties = {n: {"group": rand_group(), 
                            "fx": random.randint(1, 100),
                            "fy": random.randint(1, 100)}
                            for n in g.nodes()}

        edge_attributions = {e: {"group": rand_group()}
                            for e in g.edges()}

        nx.set_node_attributes(g, node_properties)
        nx.set_edge_attributes(g, edge_attributions)
        nx.set_node_attributes(g, {k:f"n{i}" for i, k in enumerate(g.nodes)}, "label")

        
        return g
        
    def test_spring_graph(self):
        g = self.fake_network()
        node_link_data = nx.json_graph.node_link_data(g)
        new_schema = plots.spring_force_graph(g, node_labels="label")

        self.assertEqual(
            len(vega.find_named(new_schema['data'], "node-data")['values']),  len(node_link_data["nodes"])
        )
        self.assertEqual(
            len(vega.find_named(new_schema['data'], "link-data")['values']),  len(node_link_data["links"])
        )

    def test_spring_graph_label(self):
        g = self.fake_network()
        node_link_data = nx.json_graph.node_link_data(g)
        new_schema = plots.spring_force_graph(g, node_labels=None)

        self.assertEqual(
            len(vega.find_named(new_schema['data'], "node-data")['values']),  len(node_link_data["nodes"])
        )
        self.assertEqual(
            len(vega.find_named(new_schema['data'], "link-data")['values']),  len(node_link_data["links"])
        )



class TestCalibrate(unittest.TestCase):

    def test_calibration_values(self):
        df = pd.DataFrame(list(zip([1, 2], ['test1', 'test2'], [True, True], [1, 2], [3, 4], [0, 0])),
                                              columns =['time', 'column_names', 'calibration', 'y', 'y1', 'y0'])
        
        calibration_schema = plots.contour_plot_calibration(df)

        self.assertEqual(
            len(vega.find_named(calibration_schema['data'], "table")['values']),  len(df)
        )
        self.assertEqual(
                    len(vega.find_keyed(calibration_schema["signals"], "name", "Variable")['bind']['options']), \
                          len(sorted(df["column_names"].unique().tolist()))
                )
        


        
