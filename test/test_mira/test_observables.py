import unittest
import mira
from pyciemss.PetriNetODE.base import MiraPetriNetODESystem
from pyciemss.PetriNetODE.interfaces import (
    load_petri_model,
    setup_petri_model,
    sample_petri,
    calibrate,
    load_and_calibrate_and_sample_petri_model,
    )
import pandas as pd
from pyciemss.utils.interface_utils import convert_to_output_format, csv_to_list, solutions_to_observations

import sympy
from copy import deepcopy as _d
from mira.metamodel import Observable, SympyExprStr
from mira.sources.askenet import model_from_json_file
from mira.examples.sir import sir_parameterized

class TestObservables(unittest.TestCase):
    """Test the observables of the MiraPetriNetODESystem class."""
    def setUp(self):
        """create mira observables directly."""
        self.sidarthe = load_petri_model('test/models/AMR_examples/BIOMD0000000955_askenet.json', compile_rate_law_p=True, compile_observables_p=True)
        self.observables = self.sidarthe.G.observables
        tm = _d(sir_parameterized)
        symbols = set(tm.initials)
        expr = sympy.Add(*[sympy.Symbol(s) for s in symbols])
        tm.observables = {'half_population': Observable(
            name='half_population',
            expression=SympyExprStr(expr/2))
        }
        self.sir = mira.modeling.Model(tm)
        
    def test_observables(self):
        """Test the observables of the MiraPetriNetODESystem class."""
        num_samples = 2
        data_path = 'test/test_mira/sir_data.csv'
        timepoints = [0.1, 0.2, 0.3]
        raw_sir = load_petri_model(self.sir, compile_observables_p=True)
        sir = setup_petri_model(raw_sir, 0.0, dict(susceptible_population=1000.0, infected_population=1.0, immune_population=0.0))
        sir_samples = sample_petri(sir, timepoints , num_samples)
        sir_sample_df = convert_to_output_format(sir_samples, timepoints)
        observations = solutions_to_observations(timepoints, sir_sample_df.set_index(['timepoint_id', 'sample_id']))
        observations[0].to_csv(data_path, index=False)
        sir_data = csv_to_list(data_path)
        for timepoint, data in sir_data:
            data['half_population'] = (
                data['immune_population'] + data['susceptible_population'] + data['infected_population'])/2
        inferred_parameters = calibrate(sir, sir_data, num_iterations=10)
        self.assertTrue(inferred_parameters)
        calibrated_samples = sample_petri(sir, timepoints, num_samples, inferred_parameters)
        self.assertTrue(isinstance(calibrated_samples, dict))

        unobserved_data = csv_to_list(data_path)                     
        for timepoint, data in unobserved_data:
            data['definitely_not_half_population'] = (
                data['immune_population'] + data['susceptible_population'] + data['infected_population'])/2
        with self.assertRaises(KeyError):
            inferred_parameters = calibrate(sir, unobserved_data, num_iterations=10)

        
    def test_observables_actually_calibrate(self):
        """Test the observables of the MiraPetriNetODESystem class actually generate a posterior."""
        sidarthe_data_path = 'test/test_mira/sidarthe_data.csv'
        sidarthe_model_path = 'test/models/AMR_examples/SIDARTHE.amr.json'
        sidarthe_mira = model_from_file(sidarthe_model_path)
        sidarthe_samples = load_and_sample_petri_model(sidarthe_mira, sidarthe_data_path, num_samples=1)
        sidarthe_calibrated_samples = load_and_calibrate_and_sample_petri_model(sidarthe_model_path, sidarthe_data_path, num_samples=100
                                                                 , timepoints=[0.1, 0.2, 0.3]


                                                                                                                        self.assertTrue(isinstance(sidarthe_calibrated_samples, pd.DataFrame))
        
    
        
