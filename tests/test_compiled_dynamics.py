import json
import tempfile

import mira
import networkx as nx
import pytest
import requests
import torch
from chirho.dynamical.handlers.solver import TorchDiffEq
from mira.sources.amr import model_from_url
from pyro.infer.inspect import get_dependencies

from pyciemss.compiled_dynamics import CompiledDynamics
from pyciemss.mira_integration.distributions import sort_mira_dependencies

from .fixtures import (
    ACYCLIC_MODELS,
    CYCLIC_MODELS,
    END_TIMES,
    MODEL_URLS,
    START_TIMES,
    check_is_state,
)


@pytest.mark.parametrize("url", MODEL_URLS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
def test_compiled_dynamics_load_url(url, start_time, end_time):
    """
    Test that CompiledDynamics can be loaded from a URL and that it can be simulated.

    This test verifies the following:
    - An CompiledDynamics model can be loaded from a URL and is of the correct type.
    - The model can be simulated using the simulate function.
    - The simulation result is of the correct type (torch.Tensor).
    """
    model = CompiledDynamics.load(url)
    assert isinstance(model, CompiledDynamics)

    with TorchDiffEq():
        simulation = model(torch.as_tensor(start_time), torch.as_tensor(end_time))
    check_is_state(simulation, torch.Tensor)


@pytest.mark.parametrize("url", MODEL_URLS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
def test_compiled_dynamics_load_path(url, start_time, end_time):
    res = requests.get(url)
    model_json = res.json()

    with tempfile.NamedTemporaryFile(mode="w+") as tf:
        json.dump(model_json, tf)
        tf.seek(0)
        model = CompiledDynamics.load(tf.name)
    assert isinstance(model, CompiledDynamics)

    with TorchDiffEq():
        simulation = model(torch.as_tensor(start_time), torch.as_tensor(end_time))
    check_is_state(simulation, torch.Tensor)


@pytest.mark.parametrize("url", MODEL_URLS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
def test_compiled_dynamics_load_json(url, start_time, end_time):
    res = requests.get(url)
    model_json = res.json()
    assert isinstance(model_json, dict)

    model = CompiledDynamics.load(model_json)
    assert isinstance(model, CompiledDynamics)

    with TorchDiffEq():
        simulation = model(torch.as_tensor(start_time), torch.as_tensor(end_time))
    check_is_state(simulation, torch.Tensor)


@pytest.mark.parametrize("acyclic_model", ACYCLIC_MODELS)
@pytest.mark.parametrize("cyclic_model", CYCLIC_MODELS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
def test_hierarchical_compiled_dynamics(acyclic_model, cyclic_model, start_time, end_time):
    """
    Test the loading and dependency analysis of hierarchical compiled dynamics models.

    This test verifies the following:
    - An acyclic MIRA model can be loaded from a URL and is of type TemplateModel.
    - The dependencies of the acyclic MIRA model are sorted as expected.
    - Attempting to sort dependencies for a cyclic MIRA model raises a NetworkXUnfeasible exception.
    - A CompiledDynamics model can be loaded from an acyclic URL and is of the correct type.
    - The prior and posterior dependencies of the CompiledDynamics model match the expected structure
      when given the specified start and end times.
    """
    acyclic_mira_model = model_from_url(acyclic_model.url)
    assert isinstance(acyclic_mira_model, mira.metamodel.TemplateModel)
    assert sort_mira_dependencies(acyclic_mira_model) == [
        "beta_mean",
        "gamma_mean",
        "beta",
        "gamma",
    ]
    with pytest.raises(nx.NetworkXUnfeasible):
        cyclic_mira_model = model_from_url(cyclic_model.url)
        print(cyclic_model.url)
        sort_mira_dependencies(cyclic_mira_model)
    model = CompiledDynamics.load(acyclic_model.url)
    assert isinstance(model, CompiledDynamics)
    assert get_dependencies(model, model_args=(start_time, end_time)) == {
        "prior_dependencies": {
            "persistent_beta_mean": {"persistent_beta_mean": set()},
            "persistent_gamma_mean": {"persistent_gamma_mean": set()},
            "persistent_beta": {"persistent_beta": set(), "persistent_beta_mean": set},
            "persistent_gamma": {
                "persistent_gamma": set(),
                "persistent_gamma_mean": set(),
            },
        },
        "posterior_dependencies": {
            "persistent_beta_mean": {"persistent_beta_mean": set()},
            "persistent_gamma_mean": {"persistent_gamma_mean": set()},
            "persistent_beta": {"persistent_beta": set(), "persistent_beta_mean": set},
            "persistent_gamma": {
                "persistent_gamma": set(),
                "persistent_gamma_mean": set(),
            },
        },
    }
