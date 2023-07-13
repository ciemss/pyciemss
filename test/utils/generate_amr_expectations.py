import json
import argparse
import dataclasses
import warnings
import traceback
from urllib.request import urlopen

from pyciemss.PetriNetODE.interfaces import (
    load_petri_model,
    load_and_sample_petri_model,
    load_and_calibrate_and_sample_petri_model,
)
from pyciemss.PetriNetODE.base import ScaledBetaNoisePetriNetODESystem
import pandas as pd
from pathlib import Path


AMR_URL_TEMPLATE = "https://raw.githubusercontent.com/DARPA-ASKEM/experiments/main/thin-thread-examples/mira_v2/biomodels/{biomodel_id}/model_askenet.json"
AMR_ROOT = Path("..") / "models" / "AMR_examples" / "biomodels"

biomodels = [
    "BIOMD0000000249",
    "BIOMD0000000716",
    "BIOMD0000000949",
    "BIOMD0000000956",
    "BIOMD0000000960",
    "BIOMD0000000964",
    "BIOMD0000000971",
    "BIOMD0000000976",
    "BIOMD0000000979",
    "BIOMD0000000982",
    "BIOMD0000000988",
    "MODEL1008060000",
    "MODEL1805230001",
    "MODEL2111170001",
    "BIOMD0000000294",
    "BIOMD0000000717",
    "BIOMD0000000950",
    "BIOMD0000000957",
    "BIOMD0000000962",
    "BIOMD0000000969",
    "BIOMD0000000972",
    "BIOMD0000000977",
    "BIOMD0000000980",
    "BIOMD0000000983",
    "BIOMD0000000991",
    "MODEL1008060002",
    "MODEL1808280006",
    "BIOMD0000000715",
    "BIOMD0000000726",
    "BIOMD0000000955",
    "BIOMD0000000958",
    "BIOMD0000000963",
    "BIOMD0000000970",
    "BIOMD0000000974",
    "BIOMD0000000978",
    "BIOMD0000000981",
    "BIOMD0000000984",
    "BIOMD0000001045",
    "MODEL1805220001",
    "MODEL1808280011",
]


def test_load_biomodels(config):
    try:
        model = load_petri_model(config.source_file, compile_rate_law_p=True)
        config.assertIsNotNone(model)
        config.assertIsInstance(model, ScaledBetaNoisePetriNetODESystem)
        config.tests_pass.append("load_petri_model")
    except AttributeError as e:
        warnings.warn(f"{config.id} {str(e)}")
        tb = traceback.format_exc()
        warnings.warn(tb)
    except KeyError as k:
        warnings.warn(f"{config.id} {str(k)}")
        config.tests_fail.append("load_petri_model")

    try:
        model = load_and_sample_petri_model(
            config.source,
            compile_rate_law_p=True,
            num_samples=2,
            timepoints=[0.0, 0.1, 0.2, 0.3],
        )
        config.assertIsNotNone(model)
        config.assertIsInstance(model, pd.DataFrame)
        config.tests_pass.append("load_and_sample_petri_model")
    except Exception as e:
        config.tests_fail.append("load_and_sample_petri_model")
        warnings.warn(f"{config.id} {str(e)}")
        tb = traceback.format_exc()
        warnings.warn(tb)

    try:
        with open(config.source_file, "r") as f:
            file_content = json.load(f)
            file_str = json.dumps(file_content, indent=2)

        f = urlopen(config.source_url)
        url_content = json.load(f)
        url_str = json.dumps(url_content, indent=2)
        config.assertEqual(file_str, url_str)
        config.tests_pass.append("url_against_file")
    except Exception as e:
        config.tests_fail.append("url_against_file")
        warnings.warn(f"{config.id} {str(e)}")
        tb = traceback.format_exc()
        warnings.warn(tb)


# ---------------------------
@dataclasses.dataclass(init=False)
class TestConfig:
    id: str
    source_url: str
    source_file: Path
    tests_pass: list
    tests_fail: list

    def __init__(self, id):
        self.id = id
        self.source_url = AMR_URL_TEMPLATE.format(biomodel_id=id)
        self.source_file = str(AMR_ROOT / id / "model_askenet.json")
        self.tests_pass = list()
        self.tests_fail = list()


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


parser = argparse.ArgumentParser(description="Generate expected results JSON file.")
parser.add_argument(
    "target",
    help="Where to write the expectations to. WILL NOT OVERWRITE.",
    default="AMR_expectations.json",
    type=Path,
)
parser.add_argument(
    "-r",
    "--reference",
    help="Prepare a diff against this file.",
    default="AMR_expectations_old.json",
    type=Path,
)
parser.add_argument(
    "--test",
    help="Only process <test> models, skips early file test",
    default=None,
    type=int,
)
args = parser.parse_args()

if not args.test and args.target.exists():
    raise ValueError(
        "Target file exists before run. Move/rename/delete before executing."
    )

tests = [TestConfig(model_id) for model_id in biomodels][: args.test]

for config in tests:
    test_load_biomodels(config)

try:
    with open(args.target, "x") as f:
        json.dump(tests, f, cls=JSONEncoder, indent=3)
except Exception:
    print(json.dumps(tests, cls=JSONEncoder, indent=3))
    raise Exception

# TODO: Prepare diff report
