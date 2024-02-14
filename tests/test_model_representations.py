import pytest
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


repo_root = "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main"
model_files = [
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


@pytest.mark.parametrize("model_file", model_files)
def test_representations(model_file):
    try:
        model = requests.get(model_file).json()
    except BaseException:
        assert False, f"Could not load model {model_file}"
    inventory = model_inventory.check_amr(model, summary=True)

    assert is_satisfactory(inventory), f"{model_file} not satisfactory"
