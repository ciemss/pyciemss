import json
import tempfile

import pytest
import requests
import torch
from chirho.dynamical.ops import State

from pyciemss.ensemble.compiled_dynamics import EnsembleCompiledDynamics

from .fixtures import END_TIMES, MODEL_URLS, START_TIMES


@pytest.mark.parametrize("url", MODEL_URLS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
def test_compiled_dynamics_load_url(url, start_time, end_time):
    urls = [url, url]
    model = EnsembleCompiledDynamics.load(urls, torch.ones(len(urls)), [lambda x: x, lambda x: 2 * x])
    assert isinstance(model, EnsembleCompiledDynamics)

    simulation = model(torch.as_tensor(start_time), torch.as_tensor(end_time))
    assert isinstance(simulation, State)
