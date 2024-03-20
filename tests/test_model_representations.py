import pytest
import requests

from pyciemss.compiled_dynamics import CompiledDynamics


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

    true_checks = {
        "parameter distribution exists": "Not all expected parameter distributions found",
        "parameter dist/value set": "Not all expected parameter values set",
        "rate laws present": "Not all expected rate laws found",
        "rate law vars defined": "Not all expected raw law variables found",
        "initial values present": "Not all expected initial values found",
    }

    CompiledDynamics.check_model(model, must_be_true=true_checks)


# TODO:  Need to do some tests for BAD models to see if it raises the right exception
