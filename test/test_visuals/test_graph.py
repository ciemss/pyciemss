import unittest
import pandas as pd
import xarray as xr
import numpy as np
import json
import torch

from pathlib import Path

from pyciemss.visuals import plots, vega, graphs, calibration, barycenter, trajectories
import networkx as nx
import random

_data_root = Path(__file__).parent.parent / "data"


def rand_label():
    """Get random number"""
    return random.randint(1, 10)

def rand_group():
    """get random number"""
    possible = "TUVWXYZ"
    return random.sample(possible, 1)[0]

class TestGraphs(unittest.TestCase):
    """test functions in graph"""
    def test_graph_attribute(self):
        """test new data is added to schema"""
        g = nx.generators.barabasi_albert_graph(5, 3)
        node_properties = {n: {"attribution": rand_group(), 
                            "label": rand_label()}
                            for n in g.nodes()}

        edge_attributions = {e: {"attribution": rand_group()} for e in g.edges()}

        nx.set_node_attributes(g, node_properties)
        nx.set_edge_attributes(g, edge_attributions)

        gjson = nx.json_graph.node_link_data(g)

        new_schema = graphs.attributed_graph(g)

        self.assertEqual(
            len(vega.find_named(new_schema['data'], "node-data")['values']),  len(gjson["nodes"])
        )
        self.assertEqual(
            len(vega.find_named(new_schema['data'], "link-data")['values']),  len(gjson["links"])
        )



    def test_node_missing_attribute(self):
        """test value error if node attribute missing"""
        with self.assertRaises(ValueError):
                g = nx.generators.barabasi_albert_graph(5, 3)
                node_properties = {n: {"label": rand_label()}
                                    for n in g.nodes()}

                edge_attributions = {e: {"attribution": rand_group()} for e in g.edges()}

                nx.set_node_attributes(g, node_properties)
                nx.set_edge_attributes(g, edge_attributions)
                schema = graphs.attributed_graph(g)

    def test_edge_missing_attribute(self):
        """test value error if edge attribute missing"""
        with self.assertRaises(ValueError):
                g = nx.generators.barabasi_albert_graph(5, 3)
                node_properties = {n: {"label": rand_label()}
                                    for n in g.nodes()}
                nx.set_node_attributes(g, node_properties)
                schema = graphs.attributed_graph(g)


class TestSpringGraph(unittest.TestCase):
    def fake_network(self):
        """fake graph network for testing"""
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
        
    def test_spring_graph_label(self):
        """check data is being replaced with function with label"""
        g = self.fake_network()
        node_link_data = nx.json_graph.node_link_data(g)
        new_schema = graphs.spring_force_graph(g, node_labels="label")

        self.assertEqual(
            len(vega.find_named(new_schema['data'], "node-data")['values']),  len(node_link_data["nodes"])
        )
        self.assertEqual(
            len(vega.find_named(new_schema['data'], "link-data")['values']),  len(node_link_data["links"])
        )

    def test_spring_graph(self):
        """test value error if node attribute missing"""
        g = self.fake_network()
        node_link_data = nx.json_graph.node_link_data(g)
        new_schema = graphs.spring_force_graph(g, node_labels=None)

        self.assertEqual(
            len(vega.find_named(new_schema['data'], "node-data")['values']),  len(node_link_data["nodes"])
        )
        self.assertEqual(
            len(vega.find_named(new_schema['data'], "link-data")['values']),  len(node_link_data["links"])
        )


class TestCalibration(unittest.TestCase):

    def test_calibration_assert(self):
        """check value error if creat with missing test column"""
        with self.assertRaises(ValueError):
            df = pd.DataFrame(list(zip(['test1', 'test2'], [True, True], [1, 2], [3, 4], [0, 0])),
                                                columns =['column_names', 'calibration', 'y', 'y1', 'y0'])
            
            calibration.calibration(df)





    def test_calibration_values(self):
        """test df is added to schema for calibation"""
        df = pd.DataFrame(list(zip([1, 3], ['test1', 'test2'], [True, True], [1, 2], [3, 4], [0, 0])),
                                                columns =['time', 'column_names', 'calibration', 'y', 'y1', 'y0'])
            
        calibration_schema = calibration.calibration(df)

        self.assertEqual(
            len(vega.find_named(calibration_schema['data'], "table")['values']),  len(df)
        )
        self.assertEqual(
                    len(vega.find_keyed(calibration_schema["signals"], "name", "Variable")['bind']['options']), \
                          len(sorted(df["column_names"].unique().tolist()))
                )
        

class Barycenter(unittest.TestCase):
    def test_triangle_weights(self):
        """check that creating triangle with empty top corners"""
        sample = torch.from_numpy(np.array([[1/3, 1/3, 1/3], [1/3, 1/3, 1/3], [1/3, 1/3, 1/3]]))

        json_dict = barycenter.triangle_weights(sample)
        # values will be shaped into a square with  triangle weights key outputs width and height
        self.assertEqual(json_dict["width"] * json_dict['height'], len(json_dict['values']))
        # since triangle from squal, top corners should be zero
        self.assertEqual(json_dict['values'][json_dict["width"]-1], 0)
        self.assertEqual(json_dict['values'][0], 0)
    
    def test_triangle_weights(self):
        """check title is replaces and contours are hidden"""
        sample = torch.from_numpy(np.array([[1/3, 1/3, 1/3], [1/3, 1/3, 1/3], [1/3, 1/3, 1/3]]))
        triangle_schema = plots.triangle_contour(sample, title = "test", contour=False)
        self.assertEqual(triangle_schema['title'], "test")
        self.assertEqual(vega.find_keyed(triangle_schema["marks"], "name", "_contours")\
                         ["encode"]["enter"]["strokeWidth"]['value'], 0)



        

