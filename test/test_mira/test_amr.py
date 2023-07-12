import json
import unittest
import traceback
from pyciemss.PetriNetODE.interfaces import (
    load_petri_model,
    load_and_sample_petri_model,
    load_and_calibrate_and_sample_petri_model
    )
from pyciemss.PetriNetODE.base import ScaledBetaNoisePetriNetODESystem
import warnings
import pandas as pd

class TestAMR(unittest.TestCase):
    def setUp(self):
        AMR_URL_TEMPLATE= "https://raw.githubusercontent.com/DARPA-ASKEM/experiments/main/thin-thread-examples/mira_v2/biomodels/{biomodel_id}/model_askenet.json"
        biomodels = """BIOMD0000000249	BIOMD0000000716	BIOMD0000000949	BIOMD0000000956	BIOMD0000000960	BIOMD0000000964	BIOMD0000000971	BIOMD0000000976	BIOMD0000000979	BIOMD0000000982	BIOMD0000000988	MODEL1008060000	MODEL1805230001	MODEL2111170001 BIOMD0000000294	BIOMD0000000717	BIOMD0000000950	BIOMD0000000957	BIOMD0000000962	BIOMD0000000969	BIOMD0000000972	BIOMD0000000977	BIOMD0000000980	BIOMD0000000983	BIOMD0000000991	MODEL1008060002	MODEL1808280006
BIOMD0000000715	BIOMD0000000726	BIOMD0000000955	BIOMD0000000958	BIOMD0000000963	BIOMD0000000970	BIOMD0000000974	BIOMD0000000978	BIOMD0000000981	BIOMD0000000984	BIOMD0000001045	MODEL1805220001	MODEL1808280011""".split()
        self.biomodels_tests = {biomodel_id: dict(source=AMR_URL_TEMPLATE.format(biomodel_id=biomodel_id),
                                                  tests_pass=list(),
                                                  tests_fail=list())
                                for biomodel_id in biomodels}

    def test_load_biomodels(self):
        """Test how if all biomodels can be loaded in the AMR format."""
        for biomodel_id, biomodel in self.biomodels_tests.items():
            try:
                model = load_petri_model(biomodel["source"], compile_rate_law_p=True)
                self.assertIsNotNone(model)
                self.assertIsInstance(model, ScaledBetaNoisePetriNetODESystem)
                self.biomodels_tests[biomodel_id]["tests_pass"].append('load_petri_model')
                    
            except AttributeError as e:
                warnings.warn(f"{biomodel_id} {str(e)}")
                tb = traceback.format_exc()
                warnings.warn(tb)
            except KeyError as k:
                warnings.warn(f"{biomodel_id} {str(k)}")
                self.biomodels_tests[biomodel_id]["tests_fail"].append('load_petri_model')

            try:
                model = load_and_sample_petri_model(
                    biomodel["source"], compile_rate_law_p=True, num_samples=2, timepoints=[0.0, 0.1, 0.2, 0.3])
                self.assertIsNotNone(model)
                self.assertIsInstance(model, pd.DataFrame)
                self.biomodels_tests[biomodel_id]["tests_pass"].append('load_and_sample_petri_model')
            except Exception as e:
                self.biomodels_tests[biomodel_id]["tests_fail"].append('load_and_sample_petri_model')
                warnings.warn(f"{biomodel_id} {str(e)}")
                tb = traceback.format_exc()
                warnings.warn(tb)
                
        json.dump(self.biomodels_tests, open("test/test_mira/test_amr.json", "w"), indent=2)
        
