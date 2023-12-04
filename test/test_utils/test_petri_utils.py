

import unittest
import os
import pyro
import torch


from pyciemss.utils.interface_utils import csv_to_list
import pyciemss.utils.petri_utils as petri_utils
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import networkx as nx

import json

create_png_plots = True

save_png = (
    Path(__file__).parent.parent /  "test_utils" / "reference_images" 
)

class TestPetri(unittest.TestCase):
    """Tests for the Petri Utils."""

    # Setup for the tests
    def setUp(self):
        self.petri_file = "test/test_utils/BIOMD0000000971_petri_orig.json"
        self.G = petri_utils.load(self.petri_file)
        self.url = 'https://raw.githubusercontent.com/indralab/mira/main/notebooks/evaluation_2023.01/scenario2_sidarthe_mira.json'

    #certificate error SSL   
    # def test_convert_mira_template_to_askenet_json(self):
    #     '''test returns dictionary object and
    #     keys 'states', 'transitions', 'parameters' in the model keys'''
    #     model_json = petri_utils.convert_mira_template_to_askenet_json(self.url)
    #     self.assertTrue(isinstance(model_json, dict))
    #     model_keys = model_json['model'].keys()
    #     self.assertTrue(model_keys, ['states', 'transitions', 'parameters'])
    
    def test_load_file(self):
        '''test if load will load by string value'''
        G = petri_utils.load(self.petri_file)
        self.assertTrue(G.is_multigraph())

    def test_load_file_url(self):
        '''test if load will load by url value'''
        G = petri_utils.load(self.url)
        self.assertTrue(G.is_multigraph())
    
    def test_load_file_url(self):
        '''test if load will load with json file'''
        G = petri_utils.load('test/test_utils/petri_file.json')
        self.assertTrue(G.is_multigraph())

    def test_seq_id_suffix(self):
        '''test if seq_id_suffix will add suffix when index repeats'''
        df = pd.DataFrame([1, 1])
        new_df = petri_utils.seq_id_suffix(df)
        unique_index = np.unique(new_df.index.to_list())
        self.assertEqual(len(unique_index), len(df))
                      
                    
    # function returned by petri_to_ode has G undefined
    # def test_petri_to_ode(self):
    #     '''testing '''
    #     G_state = petri_utils.add_state_indicies(self.G)
    #     function_petri = petri_utils.petri_to_ode(G_state)
    #     # function missing G variable
    #     function_petri(10, [0, 3, 2, 4, 1, 5, 6, 7])


    def test_order_state(self):
        '''
        returns states with new integer keys
        i.e.
        input states =  {'Susceptible': 0, 'Exposed': 1} returns
        (0, 1)'''
        G_state = petri_utils.add_state_indicies(self.G)
        states = {'Susceptible': 0, 'Exposed': 1}
        new_order = petri_utils.order_state(G_state, **states)
        self.assertEqual(list(new_order), list(states.values()))
                                                      

    def test_unorder_state(self):
        '''
        returns states with new integer keys
        i.e.
        input states = [0, 3, 2, 4, 1, 5, 6, 7] returns
        {'Susceptible': 0, 'Exposed': 3, 'Infected': 2, 
        'Asymptomatic': 4, 'Susceptible_quarantined': 1, 'Hospitalised': 5, 'Recovered': 6, 'Exposed_quarantined': 7}'''
        G_state = petri_utils.add_state_indicies(self.G)
        states = [0, 3, 2, 4, 1, 5, 6, 7]
        new_order = petri_utils.unorder_state(G_state, *states)
        self.assertEqual(list(new_order.values()), states)
                                                       
    
    def test_natural_order(self):
        '''check that add_state_indices return a new node attribute
        that matches with the natural order function output
        both new_order and new_index should be in form Dict[str, T]
        i.e. {'Susceptible': 0, 'Exposed': 1, 'Infected': 2}
        '''
        new_order = petri_utils.natural_order(self.G)
        G_state_idx = petri_utils.add_state_indicies(self.G)
        new_index =  nx.get_node_attributes(G_state_idx, "state_index")
        self.assertEqual(new_order, new_index)

    def test_add_state_indicies(self):
        '''check keys of state indices are all integers '''
        G_state = petri_utils.add_state_indicies(self.G)
        new_indixes = list(nx.get_node_attributes(G_state, "state_index").values())
        
        self.assertTrue(all(isinstance(x, int) for x in new_indixes))

    #TODO should this be multipliying the nodes and edges
    def test_duplicate_petri_net(self):
        'check new networkx is duplicate of input networkx'
        dup_G = petri_utils.duplicate_petri_net(self.G)
        self.assertEqual(2*len(self.G.nodes), len(dup_G.nodes))
        self.assertEqual(2*len(self.G.edges), len(dup_G.edges))


