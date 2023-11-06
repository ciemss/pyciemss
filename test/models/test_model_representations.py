import requests
import unittest
import json
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
        cls.model_files = [
            "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/flux_typed.json",
            "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/flux_typed_aug.json",
            "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/halfar.json",
            # "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/on_pop_vax.json",
            "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/sir.json",
            "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/sir_flux_span.json",
            "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/sir_typed.json",
            "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/sir_typed_aug.json",
            "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/regnet/examples/lotka_volterra.json",
            "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/stockflow/examples/sir.json",
        ]

        cls.models = {}
        for file in cls.model_files:
            try:
                cls.models[file] = requests.get(file).json()
            except Exception:
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
