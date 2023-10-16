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


class TestGraphs(unittest.TestCase):
    def rand_attributions(self):
        possible = "ABCD"
        return random.sample(possible, random.randint(1, len(possible)))

    def rand_label(self):
        return random.randint(1, 10)
    
    def test_graph_attribute(self):
        g = nx.generators.barabasi_albert_graph(5, 3)
        node_properties = {n: {"attribution": self.rand_attributions(), 
                            "label": self.rand_label()}
                            for n in g.nodes()}

        edge_attributions = {e: {"attribution": self.rand_attributions()} for e in g.edges()}

        nx.set_node_attributes(g, node_properties)
        nx.set_edge_attributes(g, edge_attributions)
        gjson = nx.json_graph.node_link_data(g)
        schema = graphs.attributed_graph(g)
        new_schema = plots.attributed_graph(g)
        self.assertEqual(
            len(vega.find_named(new_schema['data'], "node-data")['values']),  len(gjson["nodes"])
        )
        self.assertEqual(
            len(vega.find_named(new_schema['data'], "link-data")['values']),  len(gjson["links"])
        )

    def test_edge_missing_attribute(self):
        with self.assertRaises(ValueError):
                g = nx.generators.barabasi_albert_graph(5, 3)
                node_properties = {n: {"attribution": self.rand_attributions(), 
                                    "label": self.rand_label()}
                                    for n in g.nodes()}
                nx.set_node_attributes(g, node_properties)
                schema = graphs.attributed_graph(g)


    def test_node_missing_attribute(self):
        with self.assertRaises(ValueError):
                g = nx.generators.barabasi_albert_graph(5, 3)
                node_properties = {n: {"label": self.rand_label()}
                                    for n in g.nodes()}

                edge_attributions = {e: {"attribution": self.rand_attributions()} for e in g.edges()}

                nx.set_node_attributes(g, node_properties)
                nx.set_edge_attributes(g, edge_attributions)
                schema = graphs.attributed_graph(g)



class TestSpringGraph(unittest.TestCase):
    def rand_attributions(self):
        possible = "ABCD"
        return random.sample(possible, random.randint(1, len(possible)))

    def rand_label(self):
        return random.randint(1, 10)
    
    def test_graph_attribute(self):
        g = nx.generators.barabasi_albert_graph(5, 3)
        def rand_group():
            possible = "TUVWXYZ"
            return random.sample(possible, 1)[0]

        node_properties = {n: {"group": rand_group(), 
                            "fx": random.randint(1, 100),
                            "fy": random.randint(1, 100)}
                            for n in g.nodes()}

        edge_attributions = {e: {"group": rand_group()}
                            for e in g.edges()}

        nx.set_node_attributes(g, node_properties)
        nx.set_edge_attributes(g, edge_attributions)
        nx.set_node_attributes(g, {k:f"n{i}" for i, k in enumerate(g.nodes)}, "label")

        gjson = nx.json_graph.node_link_data(g)
        schema = graphs.attributed_graph(g)
        new_schema = plots.spring_force_graph(g)
        self.assertEqual(
            len(vega.find_named(new_schema['data'], "node-data")['values']),  len(gjson["nodes"])
        )
        self.assertEqual(
            len(vega.find_named(new_schema['data'], "link-data")['values']),  len(gjson["links"])
        )

    def test_edge_missing_attribute(self):
        with self.assertRaises(ValueError):
                g = nx.generators.barabasi_albert_graph(5, 3)
                node_properties = {n: {"group": rand_group(), 
                                    "fx": random.randint(1, 100),
                                    "fy": random.randint(1, 100)}
                                    for n in g.nodes()}

                nx.set_node_attributes(g, node_properties)
                nx.set_node_attributes(g, {k:f"n{i}" for i, k in enumerate(g.nodes)}, "label")

                schema = graphs.spring_force_graph(g)


    def test_node_missing_attribute(self):
        with self.assertRaises(ValueError):
                g = nx.generators.barabasi_albert_graph(5, 3)
                node_properties = {n: {"label": self.rand_label()}
                                    for n in g.nodes()}

                edge_attributions = {e: {"attribution": self.rand_attributions()} for e in g.edges()}

                nx.set_node_attributes(g, node_properties)
                nx.set_edge_attributes(g, edge_attributions)
                schema = graphs.spring_force_graph(g)