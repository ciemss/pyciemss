        AMR_URL_TEMPLATE= "https://raw.githubusercontent.com/DARPA-ASKEM/experiments/main/thin-thread-examples/mira_v2/biomodels/{biomodel_id}/model_askenet.json"
        AMR_PATH_TEMPLATE = "test/models/AMR_examples/biomodels/{biomodel_id}/model_askenet.json"
        biomodels = """BIOMD0000000249	BIOMD0000000716	BIOMD0000000949	BIOMD0000000956	BIOMD0000000960	BIOMD0000000964	BIOMD0000000971	BIOMD0000000976	BIOMD0000000979	BIOMD0000000982	BIOMD0000000988	MODEL1008060000	MODEL1805230001	MODEL2111170001 BIOMD0000000294	BIOMD0000000717	BIOMD0000000950	BIOMD0000000957	BIOMD0000000962	BIOMD0000000969	BIOMD0000000972	BIOMD0000000977	BIOMD0000000980	BIOMD0000000983	BIOMD0000000991	MODEL1008060002	MODEL1808280006
BIOMD0000000715	BIOMD0000000726	BIOMD0000000955	BIOMD0000000958	BIOMD0000000963	BIOMD0000000970	BIOMD0000000974	BIOMD0000000978	BIOMD0000000981	BIOMD0000000984	BIOMD0000001045	MODEL1805220001	MODEL1808280011""".split()
        self.biomodels_tests = {biomodel_id: dict(source_url=AMR_URL_TEMPLATE.format(biomodel_id=biomodel_id),
                                                  source_file=AMR_PATH_TEMPLATE.format(biomodel_id=biomodel_id),
                                                  tests_pass=list(),
                                                  tests_fail=list())
                                for biomodel_id in biomodels}

    def load_tester(amr):
        """Test if biomodel can be loaded in the AMR format."""
        try:
            model = load_petri_model(amr["source_path"], compile_rate_law_p=True)
            assert model is not None
            assert isinstance(model, ScaledBetaNoisePetriNetODESystem)
            amr["tests_pass"].append('load_petri_model')
        except AttributeError as e:
            warnings.warn(f"{biomodel_id} {str(e)}")
            tb = traceback.format_exc()
            warnings.warn(tb)
        except KeyError as k:
            warnings.warn(f"{biomodel_id} {str(k)}")
            self.biomodels_tests[biomodel_id]["tests_fail"].append('load_petri_model')

        try:
            samples = load_and_sample_petri_model(
                amr["source"], compile_rate_law_p=True, num_samples=2, timepoints=[0.0, 0.1, 0.2, 0.3])
            assert samples is not None
            assert isinstance(samples, pd.DataFrame)
            amr["tests_pass"].append('load_and_sample_petri_model')
        except Exception as e:
            amr["tests_fail"].append('load_and_sample_petri_model')
            warnings.warn(f"{amr['id']} {str(e)}")
            tb = traceback.format_exc()
            warnings.warn(tb)

        try:
            with open(biomodel["source_file"], "r") as f:
                file_content = json.load(f)
                file_str = json.dumps(file_content, indent=2)
            with open(biomodel["source_url"], "r") as f:
                url_content = json.load(f)
                url_str = json.dumps(url_content, indent=2)
            assert  file_str == url_str
            self.biomodels_tests[biomodel_id]["tests_pass"].append('url_against_file')
        except Exception as e:
            self.biomodels_tests[biomodel_id]["tests_fail"].append('url_against_file')
            warnings.warn(f"{biomodel_id} {str(e)}")
            tb = traceback.format_exc()
            warnings.warn(tb)
        return amr

        
