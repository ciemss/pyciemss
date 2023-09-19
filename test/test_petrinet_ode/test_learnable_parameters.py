import unittest
import os

import sympy
from mira.metamodel import (
    Concept,
    ControlledConversion,
    Initial,
    NaturalConversion,
    Parameter,
    TemplateModel,
    Distribution,
)

import os
from pyciemss.PetriNetODE.interfaces import (
    load_and_calibrate_and_sample_petri_model,
)

class TestLearnableParameters(unittest.TestCase):
    """Test that learnable parameters can be calibrated."""
    def setUp(self):
        """Load the model with only learnable parameters."""
        DEMO_PATH = "notebook/integration_demo/"
        self.data_path = os.path.join(DEMO_PATH, "data.csv")
        beta, gamma, S, I, R, total_population = sympy.symbols(
            "beta, gamma, S, I, R, total_population"
        )

        susceptible = Concept(
            name="S", identifiers={"ido": "0000514"}
        )
        infected = Concept(
            name="I", identifiers={"ido": "0000573"}
        )  # http://purl.obolibrary.org/obo/IDO_0000573
        recovered = Concept(name="R", identifiers={"ido": "0000592"})
        total_pop = 1.0

        S_to_I = ControlledConversion(
            controller=infected,
            subject=susceptible,
            outcome=infected,
            rate_law=beta * S * I / (S + I + R),
        )
        I_to_R = NaturalConversion(
            subject=infected, outcome=recovered, rate_law=gamma * I
        )
        self.no_distribution_parameters = TemplateModel(
            templates=[S_to_I, I_to_R],
            parameters={
                "beta": Parameter(name="beta", value=0.55),  # transmission rate
                "gamma": Parameter(name="gamma", value=0.2),  # recovery rate
            },
            initials={
                "S": (
                    Initial(concept=susceptible, value=(total_pop - 0.01))
                ),
                "I": (Initial(concept=infected, value=0.01)),
                "R": (Initial(concept=recovered, value=0)),
            },
        )
        self.beta_only_distribution = TemplateModel(
            templates=[S_to_I, I_to_R],
            parameters={
                "beta": Parameter(name="beta", value=0.55,
                                  distribution=Distribution(type='Uniform1',
                                                            parameters={'minimum': 0.5, 'maximum': 0.6})),
                # transmission rate
                "gamma": Parameter(name="gamma", value=0.2),  # recovery rate
            },
            initials={
                "S": (
                    Initial(concept=susceptible, value=(total_pop - 1))
                ),
                "I": (Initial(concept=infected, value=1)),
                "R": (Initial(concept=recovered, value=0)),
            },
        )
        self.gamma_only_distribution = TemplateModel(
            templates=[S_to_I, I_to_R],
            parameters={
                "beta": Parameter(name="beta", value=0.55),  # transmission rate
                "gamma": Parameter(name="gamma", value=0.2,
                                   distribution=Distribution(type='Uniform1',
                                                             parameters={'minimum': 0.1, 'maximum': 0.3}
                                                             )
                                   ),
            },
            initials={
                "S": (
                    Initial(concept=susceptible, value=(total_pop - 1))
                ),
                "I": (Initial(concept=infected, value=1)),
                "R": (Initial(concept=recovered, value=0)),
            },
        )
        
        self.all_distribution_parameters = TemplateModel(
            templates=[S_to_I, I_to_R],
            parameters={
                "beta": Parameter(name="beta", value=0.55,
                                  distribution=Distribution(type='Uniform1',
                                                            parameters={'minimum': 0.5, 'maximum': 0.6})),
                                  # transmission rate
                "gamma": Parameter(name="gamma", value=0.2,
                                   distribution=Distribution(type='Uniform1',
                                                             parameters={'minimum': 0.1, 'maximum': 0.3})),
                # recovery rate
            },
            initials={
                "S": (
                    Initial(concept=susceptible, value=(total_pop - 1))
                ),
                "I": (Initial(concept=infected, value=1)),
                "R": (Initial(concept=recovered, value=0)),
            },
        )
            
    
    def test_calibrate_fails_on_learned_parameters(self):
        """Test that calibrate fails when the model has only non-distribution parameters."""
        num_samples = 2
        timepoints = [0.0, 1.0, 2.0, 3.0, 4.0]
        num_timepoints = len(timepoints)


        self.assertTrue(
            type(
                load_and_calibrate_and_sample_petri_model(
                    self.beta_only_distribution,
                    self.data_path,
                    num_samples,
                    timepoints=timepoints,
                    method='euler',
                ) is dict))
        
        self.assertTrue(
            type(
                load_and_calibrate_and_sample_petri_model(
                    self.gamma_only_distribution,
                    self.data_path,
                    num_samples,
                    timepoints=timepoints,
                    method='euler',
                ) is dict))
        self.assertTrue(
            type(
                load_and_calibrate_and_sample_petri_model(
                    self.all_distribution_parameters,
                    self.data_path,
                    num_samples,
                    timepoints=timepoints,
                    method='euler',
                ) is dict))
        self.assertTrue(
            type(
                load_and_calibrate_and_sample_petri_model(
                    self.beta_only_distribution,
                    self.data_path,
                    num_samples,
                    timepoints=timepoints,
                    method='euler',
                    deterministic_learnable_parameters=['gamma'],
                ) is dict))
       
        with self.assertRaises(RuntimeError):
            load_and_calibrate_and_sample_petri_model(
                self.gamma_only_distribution,
                self.data_path,
                num_samples,
                timepoints=timepoints,
                method='euler',
                deterministic_learnable_parameters=['beta', 'gamma'],
            )
         
        with self.assertRaises(RuntimeError):
            load_and_calibrate_and_sample_petri_model(
                self.gamma_only_distribution,
                self.data_path,
                num_samples,
                timepoints=timepoints,
                method='euler',
                deterministic_learnable_parameters=['gamma'],
            )
        with self.assertRaises(RuntimeError):
            load_and_calibrate_and_sample_petri_model(
                self.all_distribution_parameters,
                self.data_path,
                num_samples,
                timepoints=timepoints,
                method='euler',
                deterministic_learnable_parameters=['beta', 'gamma'],
            )
        with self.assertRaises(RuntimeError):
            load_and_calibrate_and_sample_petri_model(
                self.no_distribution_parameters,
                self.data_path,
                num_samples,
                timepoints=timepoints,
                method='euler',
                )
        with self.assertRaises(RuntimeError):
            load_and_calibrate_and_sample_petri_model(
                self.no_distribution_parameters,
                self.data_path,
                num_samples,
                timepoints=timepoints,
                method='euler',
                deterministic_learnable_parameters=['beta', 'gamma'],
                )
            
        

        
                 
