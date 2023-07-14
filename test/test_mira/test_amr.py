import sys
import json
import unittest

from pathlib import Path
import warnings
import dataclasses
from urllib.request import urlopen
import traceback

from pyciemss.utils.interface_utils import solutions_to_observations
from pyciemss.PetriNetODE.interfaces import (
    load_petri_model,
    load_and_sample_petri_model,
    load_and_calibrate_and_sample_petri_model,
)
from pyciemss.PetriNetODE.base import MiraPetriNetODESystem
import pandas as pd


# ####  IMPORTANT USAGE NOTE #######
# This test suite uses an generated file as its "ground truth".
# To re-generate this file, this test can be run as a python script.


_config_file = Path(__file__).parent / "AMR_expectations.json"
_report_file = Path(__file__).parent / "AMR_expectations_report.json"
AMR_URL_TEMPLATE = "https://raw.githubusercontent.com/DARPA-ASKEM/experiments/main/thin-thread-examples/mira_v2/biomodels/{biomodel_id}/model_askenet.json"
AMR_ROOT = Path(__file__).parent / ".." / "models" / "AMR_examples" / "biomodels"
SYN_SAMPLE_DATA_TEMP = Path(__file__).parent / "__sample_data.csv"


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


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
    """Loads; Loads & Samples; Loads, Calibrates & Samples a model,
    Runs assertion tests against those to determine if they are passing minimum viability.
    """
    source_file = (AMR_ROOT / config.id / "model_askenet.json").absolute()
    if not Path(source_file).exists():
        config.tests_fail.append("File not found")
        return

    try:
        model = load_petri_model(str(source_file), compile_rate_law_p=True)
        context.assertIsNotNone(model)
        context.assertIsInstance(model, MiraPetriNetODESystem)

        config.tests_pass.append("load_petri_model")
    except Exception as e:
        warnings.warn(f"{config.id} {str(e)}")
        tb = traceback.format_exc()
        warnings.warn(tb)
        config.tests_fail.append("load_petri_model")
        return

    try:
        sample_timepoints = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        samples = load_and_sample_petri_model(
            str(source_file),
            compile_rate_law_p=True,
            num_samples=2,
            timepoints=sample_timepoints,
        )
        context.assertIsNotNone(samples)
        context.assertIsInstance(samples, pd.DataFrame)
        config.tests_pass.append("load_and_sample_petri_model")
    except Exception as e:
        warnings.warn(f"{config.id} {str(e)}")
        tb = traceback.format_exc()
        warnings.warn(tb)
        config.tests_fail.append("load_and_sample_petri_model")
        return

    try:
        synthetic_obs_data = solutions_to_observations(sample_timepoints, samples)
        synthetic_obs_data[0].to_csv(SYN_SAMPLE_DATA_TEMP, index=False)

        calib_sample_timepoints = sample_timepoints + [0.9, 1.0, 1.1, 1.2]
        calibrated_samples = load_and_calibrate_and_sample_petri_model(
            str(source_file),
            SYN_SAMPLE_DATA_TEMP,
            num_samples=5,
            timepoints=calib_sample_timepoints,
            num_iterations=2,
        )

        context.assertIsNotNone(calibrated_samples)
        context.assertIsInstance(calibrated_samples, pd.DataFrame)
        config.tests_pass.append("load_and_calibrate_and_sample_petri_model")
    except Exception as e:
        warnings.warn(f"{config.id} {str(e)}")
        tb = traceback.format_exc()
        warnings.warn(tb)
        config.tests_fail.append("load_and_calibrate_and_sample_petri_model")
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

            results.append(result)
            if set(result.tests_fail) != set(ref.tests_fail):
                failures.append(result)

        with open(_report_file, "w") as f:
            json.dump(results, f, cls=JSONEncoder, indent=3)

        self.assertIs(len(failures), 0, "Unexpected failures detected")


if __name__ == "__main__":
    import argparse

    class FakeTestContext:
        def assertIsNotNone(self, model):
            if model is None:
                raise AssertionError("Value is None")

        def assertIsInstance(self, model, type):
            if not isinstance(model, type):
                raise AssertionError("Value not of expected type")

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

    parser.add_argument(
        "--diff-details",
        help="Print out all differences from the reference (default only prints regressions; aka, old-pass/new-fail and missing)",
        action="store_true",
    )
    parser.add_argument("--nowarn", help="Suppress warning output", action="store_true")

    args = parser.parse_args()

    if not args.test and args.target.exists():
        raise ValueError(
            "Target file exists before run. Move/rename/delete before executing."
        )

    biomodels = [f.name for f in AMR_ROOT.glob("*")]

    tests = [Configuration(model_id) for model_id in biomodels][: args.test]

    for config in tests:
        with warnings.catch_warnings():
            if args.nowarn:
                warnings.simplefilter("ignore")
            try_loading_biomodel(config, FakeTestContext())

    try:
        with open(args.target, "x") as f:
            json.dump(tests, f, cls=JSONEncoder, indent=3)
    except FileExistsError:
        print("\n\nTarget file already exists. NOT OVERWRITTING.  Output below:")
        print(f"\tTarget file: {args.target}\n")
        print(json.dumps(tests, cls=JSONEncoder, indent=3))
        sys.exit(-1)

    # Compute summary stats
    from collections import Counter
    from itertools import chain

    failures = Counter(chain.from_iterable(t.tests_fail for t in tests))
    successes = Counter(chain.from_iterable(t.tests_pass for t in tests))
    new_fails = pd.Series(failures).rename("Fails")
    new_passes = pd.Series(successes).rename("Passes")
    stats = pd.concat([new_fails, new_passes], axis="columns")
    changes = ""

    if args.reference is not None:
        with open(args.reference) as f:
            reference = json.load(f)

        refs = [Configuration.from_json(config) for config in reference]
        failures = Counter(chain.from_iterable(t.tests_fail for t in refs))
        successes = Counter(chain.from_iterable(t.tests_pass for t in refs))
        ref_fails = pd.Series(failures).rename("Prior Fails")
        ref_passes = pd.Series(successes).rename("Prior Passes")

        ref_stats = pd.concat([ref_fails, ref_passes], axis="columns").fillna(0)
        stats = stats.join(ref_stats)[
            ["Fails", "Prior Fails", "Passes", "Prior Passes"]
        ]

        def find_named(name, collection):
            for e in collection:
                if e.id == name:
                    return e
            return None

        def diff(old_result, new_result):
            if new_result is None:
                return "Not present in new"

            old_pass = set(old_result.tests_pass)
            old_fail = set(old_result.tests_fail)
            new_pass = set(new_result.tests_pass)
            new_fail = set(new_result.tests_fail)

            result = {
                "old-pass/new-fail": [*old_pass.intersection(new_fail)]
            }  # Regression

            if args.diff_details:
                result = {
                    **result,
                    **{
                        "old-fail/new-pass": [
                            *old_fail.intersection(new_pass)
                        ],  # Imrpoved
                        "old-pass/new-pass": [
                            *old_pass.intersection(new_pass)
                        ],  # Stayed at passing
                        "old-fail/new-fail": [
                            *old_fail.intersection(new_fail)
                        ],  # Stayed bad
                    },
                }

            return {k: v for k, v in result.items() if len(v) > 0}

        changes = {old.id: diff(old, find_named(old.id, tests)) for old in refs}
        changes = {k: v for k, v in changes.items() if len(v) > 0}
        print("\n\n -- Detail regressions ---------------------------------- ")
        print(json.dumps(changes, indent=3))

        # TODO: List exact changes

    print("\n\n -- Summary Stats ---------------------------------- ")
    print(stats.fillna(0))
