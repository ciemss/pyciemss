import pytest
import requests
from askem_model_representations import model_inventory


def is_satisfactory(inventory):
    """All true/false statements in the invetory need to be true for it to be used.
    This removes teh counts
    """
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
    f"{repo_root}/petrinet/examples/ont_pop_vax.json",
    f"{repo_root}/petrinet/examples/sir_flux_span.json",
    f"{repo_root}/regnet/examples/lotka_volterra.json",
    f"{repo_root}/stockflow/examples/sir.json",
    # f"{repo_root}/petrinet/examples/halfar.json",
    # f"{repo_root}/petrinet/examples/sir.json",
    # f"{repo_root}/petrinet/examples/sir_typed.json",
    # f"{repo_root}/petrinet/examples/sir_typed_aug.json",
    (
        "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration"
        "/main/data/models/SEIRHD_NPI_Type1_petrinet.json"
    ),
]


@pytest.mark.parametrize("model_file", model_files)
def test_representations(model_file):
    try:
        model = requests.get(model_file).json()
    except BaseException:
        assert False, "Could not load model"
    inventory = model_inventory.check_amr(model, summary=True)

    keys_to_check = [
        "parameter distribution exists",
        "parameter dist/value set",
        "rate laws present",
        "rate law vars defined",
        "initial values present",
    ]

    for key in keys_to_check:
        assert inventory[key], f"'{key}' check failed in {inventory}"
