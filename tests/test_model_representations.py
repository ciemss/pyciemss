import unittest

import requests
from askem_model_representations import model_inventory


def is_satisfactory(inventory):
    inventory = inventory.copy()
    inventory.pop("source", None)
    inventory.pop("observables found", None)
    return all(inventory.values())


def simplify_inventory(inventory):
    inventory = inventory.copy()
    to_remove = ["source", "observables found"] + [k for k, v in inventory.items() if v]
    for k in to_remove:
        inventory.pop(k, None)
    return inventory


class TestModelRepresentations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Fetch model files.
        # NOTE: If 'discovery' is required, look at the python 'git' module
        repo_root = (
            "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main"
        )
        cls.model_files = [
            f"{repo_root}/petrinet/examples/flux_typed.json",
            f"{repo_root}/petrinet/examples/flux_typed_aug.json",
            # "{repo_root}/halfar.json",
            # "{repo_root}/on_pop_vax.json",
            # "{repo_root}/sir.json",
            f"{repo_root}/petrinet/examples/sir_flux_span.json",
            # "{repo_root}/sir_typed.json",
            # "{repo_root}/sir_typed_aug.json",
            f"{repo_root}/regnet/examples/lotka_volterra.json",
            f"{repo_root}/stockflow/examples/sir.json",
        ]

        cls.models = {}
        for file in cls.model_files:
            try:
                cls.models[file] = requests.get(file).json()
            except Exception:
                unittest.skip("Could not fetch all requsted model files")
                # raise ValueError(file)
                pass

    def test_representations(self):
        self.assertEqual(
            len(self.models),
            len(self.model_files),
            "Not all models loaded as expected.",
        )

        issues = {}
        for model_name, model in self.models.items():
            inventory = model_inventory.check_amr(model, summary=True)
            if not is_satisfactory(inventory):
                issues[model_name] = inventory

        self.assertFalse(issues, f"{len(issues)} issues found")
