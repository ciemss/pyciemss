import unittest
import torch
import mira

from mira.modeling.askenet.petrinet import AskeNetPetriNetModel
import mira
import requests
from mira.examples.sir import sir_parameterized as sir # MIRA model
from mira.modeling.askenet.petrinet import AskeNetPetriNetModel
from pyciemss.interfaces import DynamicalSystem, setup_model
from pyciemss.utils import reparameterize
from pyciemss.PetriNetODE.base import ScaledBetaNoisePetriNetODESystem
from pyciemss.PetriNetODE.models import SIR_with_uncertainty, SEIARHD # Hand model
from pyciemss.PetriNetODE.interfaces import (setup_model, reset_model, intervene,
                                             sample, calibrate, optimize, load_petri_model)
from pyciemss.utils import get_tspan
from mira.sources.askenet import model_from_url

class TestPetrinetDerivatives(unittest.TestCase):
    '''
    Test the MIRA model derivatives against Hand-coded model
    '''
    def setUp(self):
        """Initialize and load the Mira and Hand models."""
        self.setUp_MIRA()
        self.setUp_Hand()
        self.setup_ASKENET()
        #self.setup_amr_vs_hand()
        self.setup_SIDARTHE()
        
    def setUp_MIRA(self):
        """Load the SIR mira model and initialize it."""
        
        # Set timeframe and resolution (89 days with 890 points throughout (10x resolution))
        self.tspan = get_tspan(1, 89, 890)
        # Load petri, deterministic
        self.petri = load_petri_model(sir, add_uncertainty=True)
        #self.assertIsInstance(self.petri, "x") # test that it is "x"
        
        # Load starting values
        self.initial_state = dict(susceptible_population=3.0,
                                  infected_population=1.0,
                                  immune_population=0.0)
        
        # Initialize
        self.MIRA_model = setup_model(self.petri, start_time=0, start_state=self.initial_state)
        #self.assertIsInstance(self.MIRA_model, "x") # test that it is "x"

        
    
    def setUp_Hand(self):
        """Instantiate the SIR_with_uncertainty model and initialize it."""
        # tspan is the same
        
        # Load hardcoded function
        self.raw = SIR_with_uncertainty(N=100.0, beta=0.1, gamma=0.2) # N here unecessary, makes its own
        #self.assertIsInstance(self.raw, "x") # test that it is "x"
        
        # initial_state is the same
        
        # Initialize
        self.Hand_model = setup_model(self.raw, start_time=0, start_state=self.initial_state)
        #self.assertIsInstance(self.Hand_model, SIR_with_uncertainty) # test that it is "x"

    def setup_ASKENET(self):
        """Instantiate the ASKENET model and initialize it."""
        mira_model = model_from_url('https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/sir.json')
        self.sir_askenet = load_petri_model(mira_model)
        # Initialize
        self.askenet2hand = dict(S_sol='susceptible_population_sol',
                            I_sol='infected_population_sol',
                            R_sol='immune_population_sol')
        
        self.hand2askenet = dict(susceptible_population='S',
                            infected_population='I',
                            immune_population='R')
        
        initial_state = {self.hand2askenet[k]: v for k, v in self.initial_state.items()}

        self.ASKENET_model = setup_model(self.sir_askenet, start_time=0, start_state=initial_state)

    # def setup_amr_vs_hand(self):
    #     """Comparison between AMR and hand models."""
    #     N = 100000
    #     seiarhd_hand_model = SEIARHD(N=N, beta=0.55, delta=1.5, alpha=4, pS=0.7, gamma=0.2, hosp=0.1, los=7, dh=0.1, dnh=0.001)
    #     initial_state = seiarhd_hand_model.create_start_state_symp(N)
    #     self.seiarhd_hand_model = setup_model(seiarhd_hand_model, start_time=0, start_state=initial_state)
    #     url = 'https://raw.githubusercontent.com/ciemss/pyciemss/main/notebook/Examples_for_TA2_Model_Representation/SEIARHD_AMR.json'
    #     seiarhd_amr_model = load_petri_model(url)
    #     #initial_state = {param: amr_model.G.template_model.initials[param].value for param in amr_model.G.template_model.initials.keys()}
    #     self.seiarhd_amr_model = setup_model(seiarhd_amr_model, start_time=0, start_state=initial_state)

    def setup_SIDARTHE(self):
        """Set up the MIRA and ASKENET SIDARTHE models."""
        sidarthe_mira_url = 'test/models/april_ensemble_demo/BIOMD0000000955_template_model.json'
        sidarthe_mira_model = load_petri_model(sidarthe_mira_url)

        sidarthe_askenet_url = 'test/models/AMR_examples/BIOMD0000000955_askenet.json'
        sidarthe_askenet_model = load_petri_model(sidarthe_askenet_url)

        initial_state = {param: sidarthe_mira_model.G.template_model.initials[param].value
                         for param in sidarthe_mira_model.G.template_model.initials.keys()}
        self.sidarthe_mira_model = setup_model(sidarthe_mira_model, start_time=0, start_state=initial_state)
        self.sidarthe_askenet_model = setup_model(sidarthe_askenet_model, start_time=0, start_state=initial_state)


    def test_derivs(self):
        """Sample from the MIRA object and set the parameters of the manual object to be the same."""
        nsamples = 5
        timepoints = [1.0, 2.0, 3.0]
        prior_samples = sample(self.MIRA_model, timepoints, nsamples)
        for i in range(nsamples):
            hand_model = reparameterize(self.Hand_model, {
                'beta': prior_samples['beta'][i],
                'gamma': prior_samples['gamma'][i]
            })
            trajectories = sample(hand_model, timepoints, 1)
            for trajectory in prior_samples:
                if '_sol' in trajectory:
                    self.assertTrue(
                        torch.allclose(
                            prior_samples[trajectory][i],
                            trajectories[trajectory][0],
                            atol=1e-4
                        ),
                        f"MIRA {trajectory} trajectory {i}: {prior_samples[trajectory][i]}\n"
                        f"Hand {trajectory} trajectory: {trajectories[trajectory][0]}"
                    )

    def test_askenet(self):
        """Test the ASKENET model representation against a manual model."""
        nsamples = 5
        timepoints = [1.0, 2.0, 3.0]
        amr_trajectories = sample(self.ASKENET_model, timepoints, nsamples)
        for i in range(nsamples):
            hand_model = reparameterize(self.Hand_model, {
                'beta': amr_trajectories['beta'][i],
                'gamma': amr_trajectories['gamma'][i]
            })
            hand_trajectories = sample(hand_model, timepoints, 1)
            for state_variable in amr_trajectories:
                if '_sol' in state_variable:
                    self.assertTrue(
                        torch.allclose(
                            amr_trajectories[state_variable][i],
                            hand_trajectories[self.askenet2hand[state_variable]][0],
                            atol=1e-4
                        ),
                        f"ASKENET {state_variable} trajectory {i}: {amr_trajectories[state_variable][i]}\n"
                        f"Hand {self.askenet2hand[state_variable]} trajectory: {hand_trajectories[self.askenet2hand[state_variable]][0]}"
                    )

    def test_sidarthe(self):
        """Test mira SIDARTHE model against amr model."""
        nsamples = 5
        timepoints = [1.0, 2.0, 3.0]
        mira_trajectories = sample(self.sidarthe_mira_model, timepoints, nsamples)
        for i in range(nsamples):
            amr_model = reparameterize(self.sidarthe_askenet_model, {
                param : mira_trajectories[param][i]
                for param in mira_trajectories.keys()                 
                if '_sol' not in param and param in self.sidarthe_askenet_model.G.parameters.keys()
                }
            )
            amr_trajectories = sample(amr_model, timepoints, 1)
            for state_variable in mira_trajectories:
                if '_sol' in state_variable:
                    self.assertTrue(
                        torch.allclose(
                            mira_trajectories[state_variable][i],
                            amr_trajectories[state_variable][0],
                            atol=1e-4
                        ),
                        f"SIDARTHE Mira {state_variable} trajectory {i}: {mira_trajectories[state_variable][i]}\n"
                        f"SIDARTHE ASKENET {state_variable} trajectory: {amr_trajectories[state_variable][0]}"
                    )
                
        

    # def test_amr_vs_hand(self):
    #     """Test the ASKENET model representation against a manual model."""
    #     nsamples = 5
    #     timepoints = [1.0, 2.0, 3.0]
    #     hand_trajectories = sample(self.seiarhd_hand_model, timepoints, nsamples)
    #     for i in range(nsamples):
    #         seiarhd_amr_model = reparameterize(self.seiarhd_amr_model, {
    #             param : hand_trajectories[param][i]
    #             for param in hand_trajectories.keys()                 
    #             if '_sol' not in param
    #             }
    #         )
    #         amr_trajectories = sample(seiarhd_amr_model, timepoints, 1)
    #         for state_variable in hand_trajectories:
    #             if '_sol' in state_variable:
    #                 self.assertTrue(
    #                     torch.allclose(
    #                         hand_trajectories[state_variable][i],
    #                         amr_trajectories[state_variable][0],
    #                         atol=1e-4
    #                     ),
    #                     f"Hand {state_variable} trajectory {i}: {hand_trajectories[state_variable][i]}\n"
    #                     f"ASKENET {state_variable} trajectory: {amr_trajectories[state_variable][0]}"
    #                 )
                
        
