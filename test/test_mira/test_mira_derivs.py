import unittest
import torch
from mira.examples.sir import sir_parameterized as sir # MIRA model
from pyciemss.PetriNetODE.models import SIR_with_uncertainty # Hand model
from pyciemss.PetriNetODE.interfaces import (setup_model, reset_model, intervene,
                                             sample, calibrate, optimize, load_petri_model)
from pyciemss.utils import get_tspan

class TestInterventions(unittest.TestCase):
    '''
    Test the MIRA model derivatives against Hand-coded model
    '''
    def setUp_MIRA(self):
        
        # Set timeframe and resolution (89 days with 890 points throughout (10x resolution))
        self.tspan = get_tspan(1, 89, 890)
        
        # Load petri, deterministic
        self.petri = load_petri_model(sir, add_uncertainty=True)
        self.assertIsInstance(self.petri, "x") # test that it is "x"
        
        # Load starting values
        self.initial_state = dict(susceptible_population=3.0,
                                  infected_population=1.0,
                                  immune_population=0.0)
        
        # Initialize
        self.MIRA_model = setup_model(self.petri, start_time=0, start_state=self.initial_state)
        self.assertIsInstance(self.MIRA_model, "x") # test that it is "x"
    
    def setUp_Hand(self):
        
        # tspan is the same
        
        # Load hardcoded function
        self.raw = SIR_with_uncertainty(N=100.0, beta=0.1, gamma=0.2) # N here unecessary, makes its own
        self.assertIsInstance(self.raw, "x") # test that it is "x"
        
        # initial_state is the same
        
        # Initialize
        self.Hand_model = setup_model(self.raw, start_time=0, start_state=self.initial_state)
        self.assertIsInstance(self.Hand_model, "x") # test that it is "x"
        
    def test_derivs(self):
        # ...
        self.assertTrue(...)
