import json
import unittest
from pathlib import Path
import json
import dataclasses
from urllib.request import urlopen

from pyciemss.PetriNetODE.interfaces import (
    load_petri_model,
    load_and_sample_petri_model,
    load_and_calibrate_and_sample_petri_model,
)
from pyciemss.PetriNetODE.base import ScaledBetaNoisePetriNetODESystem
import pandas as pd
from pathlib import Path

_config_file = Path(__file__).parent / "AMR_expectations.json"
_report_file = Path(__file__).parent / "AMR_expectations_report.json"
AMR_URL_TEMPLATE = "https://raw.githubusercontent.com/DARPA-ASKEM/experiments/main/thin-thread-examples/mira_v2/biomodels/{biomodel_id}/model_askenet.json"
AMR_ROOT = Path(__file__).parent / ".." / "models" / "AMR_examples" / "biomodels"


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


# ---------------------------
@dataclasses.dataclass(init=False)
class Configuration:
    id: str
    source_url: str
    tests_pass: list
    tests_fail: list

    def __init__(self, id):
        self.id = id
        self.source_url = AMR_URL_TEMPLATE.format(biomodel_id=id)
        self.tests_pass = list()
        self.tests_fail = list()

    @staticmethod
    def from_json(spec):
        v = Configuration(spec["id"])
        v.source_url == spec["source_url"]
        v.tests_pass = spec["tests_pass"]
        v.tests_fail = spec["tests_fail"]
        return v


def try_loading_biomodel(config: Configuration, context: any):
    source_file = (AMR_ROOT / config.id / "model_askenet.json").absolute()
    if not Path(source_file).exists():
        config.tests_fail.append("File not found")
        return

    try:
        model = load_petri_model(str(source_file), compile_rate_law_p=True)
        context.assertIsNotNone(model)
        context.assertIsInstance(model, ScaledBetaNoisePetriNetODESystem)
        config.tests_pass.append("load_petri_model")
    except KeyError:
        config.tests_fail.append("load_petri_model")
        return

    try:
        model = load_and_sample_petri_model(
            str(source_file),
            compile_rate_law_p=True,
            num_samples=2,
            timepoints=[0.0, 0.1, 0.2, 0.3],
        )
        context.assertIsNotNone(model)
        context.assertIsInstance(model, pd.DataFrame)
        config.tests_pass.append("load_and_sample_petri_model")
    except Exception:
        config.tests_fail.append("load_and_sample_petri_model")
        return


class TestAMR(unittest.TestCase):
    def setUp(self):
        with open(_config_file) as f:
            self.configs = json.load(f)

    def test_remotes_changed(self):
        changes = []
        for src in self.configs:
            config = Configuration.from_json(src)
            source_file = AMR_ROOT / config.id / "model_askenet.json"

            with open(source_file, "r") as f:
                file_content = json.load(f)
                file_str = json.dumps(file_content, indent=2)

            f = urlopen(config.source_url)
            url_content = json.load(f)
            url_str = json.dumps(url_content, indent=2)

            if url_str != file_str:
                changes.append(config.id)

        if len(changes) > 0:
            print(f"Files changed: {', '.join(changes)}")

        self.assertEqual([], changes, "URLs and local source out of sync")

    def test_configs(self):
        print("Starting test...")
        print(f"ABS PATH: {AMR_ROOT.absolute()}")
        results = []
        failures = []
        for config in self.configs:
            ref = Configuration.from_json(config)
            result = Configuration.from_json(config)

            try_loading_biomodel(result, self)

            # self.assertEqual(ref, result)

            results.append(result)
            if set(result.tests_fail) != set(ref.tests_fail):
                failures.append(result)

        with open(_report_file, "w") as f:
            json.dump(results, f, cls=JSONEncoder, indent=3)

        self.assertIs(len(failures), 0, "Unexpected failures detected")
