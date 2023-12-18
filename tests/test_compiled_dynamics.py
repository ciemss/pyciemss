import json
import tempfile

import pytest
import requests
import torch
from chirho.dynamical.handlers.solver import TorchDiffEq

from pyciemss.compiled_dynamics import CompiledDynamics

from .fixtures import END_TIMES, MODEL_URLS, START_TIMES, check_is_state


@pytest.mark.parametrize("url", MODEL_URLS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
def test_compiled_dynamics_load_url(url, start_time, end_time):
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
