import json
import sys
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
AMR_ROOT = Path(__file__).parent / ".." / "models" / "AMR_examples" / "biomodels"


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


def try_loading_biomodel(config):
    try:
        source_file = str(AMR_ROOT / config.id / "model_askenet.json")
        model = load_petri_model(source_file, compile_rate_law_p=True)
        if model is None:
            config.tests_fail.append("load_petri_model")
        elif not isinstance(model, ScaledBetaNoisePetriNetODESystem):
            config.tests_fail.append("load_petri_model")
        else:
            config.tests_pass.append("load_petri_model")
    except AttributeError as e:
        warnings.warn(f"{config.id} {str(e)}")
        tb = traceback.format_exc()
        warnings.warn(tb)
    except KeyError as k:
        warnings.warn(f"{config.id} {str(k)}")
        config.tests_fail.append("load_petri_model")

    try:
        result = load_and_sample_petri_model(
            source_file,
            compile_rate_law_p=True,
            num_samples=2,
            timepoints=[0.0, 0.1, 0.2, 0.3],
        )
        if result is None:
            config.tests_fail.append("load_and_sample_petri_model")
        elif not isinstance(result, pd.DataFrame):
            config.tests_fail.append("load_and_sample_petri_model")
        else:
            config.tests_pass.append("load_and_sample_petri_model")
    except Exception as e:
        config.tests_fail.append("load_and_sample_petri_model")
        warnings.warn(f"{config.id} {str(e)}")
        tb = traceback.format_exc()
        warnings.warn(tb)

    try:
        with open(source_file, "r") as f:
            file_content = json.load(f)
            file_str = json.dumps(file_content, indent=2)

        f = urlopen(config.source_url)
        url_content = json.load(f)
        url_str = json.dumps(url_content, indent=2)

        if file_str != url_str:
            raise ValueError(f"File and URL content do not match for {config.id}")

        config.tests_pass.append("url_against_file")
    except Exception as e:
        config.tests_fail.append("url_against_file")
        warnings.warn(f"{config.id} {str(e)}")
        tb = traceback.format_exc()
        warnings.warn(tb)


if __name__ == "__main__":

    class JSONEncoder(json.JSONEncoder):
        def default(self, o):
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            return super().default(o)

    parser = argparse.ArgumentParser(description="Generate expected results JSON file.")
    parser.add_argument(
        "-t",
        "--target",
        help="Where to write the expectations to. WILL NOT OVERWRITE.",
        default=Path(".") / ".." / "test_mira" / "AMR_expectations.json",
        required=False,
        type=Path,
    )
    parser.add_argument(
        "-r",
        "--reference",
        help="Prepare a diff against this file.",
        default=None,
        required=False,
        type=Path,
    )
    parser.add_argument(
        "--test",
        help="Only process <test> models, skips early file test",
        default=None,
        required=False,
        type=int,
    )
    parser.add_argument("--quiet", help="Suppress warning output", action="store_true")

    args = parser.parse_args()

    if not args.test and args.target.exists():
        raise ValueError(
            "Target file exists before run. Move/rename/delete before executing."
        )

    biomodels = [f.name for f in AMR_ROOT.glob("*")]

    tests = [Configuration(model_id) for model_id in biomodels][: args.test]

    for config in tests:
        with warnings.catch_warnings():
            if args.quiet:
                warnings.simplefilter("ignore")
            try_loading_biomodel(config)

    try:
        with open(args.target, "x") as f:
            json.dump(tests, f, cls=JSONEncoder, indent=3)
    except FileExistsError:
        print("\n\nTarget file already exists. NOT OVERWRITTING.  Output below:")
        print(f"\tTarget file: {args.target}\n")
        print(json.dumps(tests, cls=JSONEncoder, indent=3))
        sys.exit(-1)

    if args.reference is not None:
        pass
        # TODO: Prepare diff report
