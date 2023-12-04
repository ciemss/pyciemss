

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
    """Tests for the ODE interfaces."""

    # Setup for the tests
    def setUp(self):
        self.petri_file = "test/test_utils/BIOMD0000000971_petri_orig.json"
        self.G = petri_utils.load(self.petri_file)
        self.url = 'https://raw.githubusercontent.com/indralab/mira/main/notebooks/evaluation_2023.01/scenario2_sidarthe_mira.json'
        
    def test_convert_mira_template_to_askenet_json(self):
        model_json = petri_utils.convert_mira_template_to_askenet_json(self.url)
        self.assertTrue(isinstance(model_json, dict))
        model_keys = model_json['model'].keys()
        self.assertTrue(model_keys, ['states', 'transitions', 'parameters'])
    
    def test_load_file(self):
        G = petri_utils.load(self.petri_file)
        self.assertTrue(G.is_multigraph())

    def test_load_file_url(self):
        G = petri_utils.load(self.url)
        self.assertTrue(G.is_multigraph())
    
    def test_load_file_url(self):
        G = petri_utils.load('test/test_utils/petri_file.json')
        self.assertTrue(G.is_multigraph())

    def test_seq_id_suffix(self):
        df = pd.DataFrame([1, 1])
        new_df = petri_utils.seq_id_suffix(df)
        unique_index = np.unique(new_df.index.to_list())
        self.assertEqual(len(unique_index), len(df))
                      
                    

    def test_petri_to_ode(self):
        G_state = petri_utils.add_state_indicies(self.G)
        states = {'Susceptible': 0, 'Exposed': 2, 'Infected': 1}
        petri_utils.petri_to_ode(G_state, **states)
        with self.assertRaises(TypeError):
            petri_utils.petri_to_ode(G_state, "test")


    def test_order_state(self):
        G_state = petri_utils.add_state_indicies(self.G)
        states = {'Susceptible': 0, 'Exposed': 1}
        new_order = petri_utils.order_state(G_state, **states)
        self.assertEqual(list(new_order), list(states.values()))
                                                      

    def test_unorder_state(self):
        G_state = petri_utils.add_state_indicies(self.G)
        states = [0, 3, 2, 4, 1, 5, 6, 7]
        new_order = petri_utils.unorder_state(G_state, *states)
        self.assertEqual(list(new_order.values()), states)
                                                       
    
    def test_natural_order(self):
        G = petri_utils.natural_order(self.G)
        state_idx = petri_utils.ordering_principle(G)
        new_index = nx.node_attribute("state_index")
        self.assertEqual(state_idx, new_index)

    def test_add_state_indicies(self):
        G_state = petri_utils.add_state_indicies(self.G)
        state2ind = {node: data["state_index"] for node, data in G_state.nodes(data=True)
                 if data["type"] == "state"}
        self.assertEqual(list(state2ind.keys()), list(range(0,8)))


    def test_duplicate_petri_net(self):
        dup_G = petri_utils.duplicate_petri_net(self.G)
        self.assertEqual(self.G, dup_G)

    def test_intervene_petri_net(self):
        petri_utils.intervene_petri_net(self.G, self.G)


#         from mira.modeling.askenet.petrinet import AskeNetPetriNetModel
# from mira.sources.askenet.petrinet import template_model_from_askenet_json

# mira_template_url = 'https://raw.githubusercontent.com/indralab/mira/main/notebooks/evaluation_2023.01/scenario2_sidarthe_mira.json'
# #url = 'https://raw.githubusercontent.com/indralab/mira/main/notebooks/evaluation_2023.01/scenario1_sir_mira.json'
# res = requests.get(mira_template_url)
# model_json = res.json()
# mira_template_model = mira.metamodel.TemplateModel.from_json(model_json)
# mira_model = mira.modeling.Model(mira_template_model)
# askenet_model = AskeNetPetriNetModel(mira_model)
# askenet_mira_template_model = template_model_from_askenet_json(askenet_model.to_json())

# uncertain_lotka_volterra = setup_petri_model(
#     raw_lotka_volterra, start_time=0.0, start_state=dict(
#         Rabbits=1.0, Wolves=1.0))

# observed_lotka_volterra = reparameterize(
#     uncertain_lotka_volterra,
#     dict(alpha=0.67,beta=1.33,gamma=1.0, delta = 1.0))