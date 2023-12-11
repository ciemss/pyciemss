import pytest
import torch
from chirho.dynamical.handlers.solver import TorchDiffEq

from pyciemss.ensemble.compiled_dynamics import EnsembleCompiledDynamics

from .fixtures import END_TIMES, MODEL_URLS, START_TIMES, check_is_state


@pytest.mark.parametrize("url", MODEL_URLS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
def test_ensemble_compiled_dynamics_load_url(url, start_time, end_time):
    urls = [url, url]
    model = EnsembleCompiledDynamics.load(
        urls,
        torch.ones(len(urls)),
        [lambda x: x, lambda x: {k: 2 * v for k, v in x.items()}],
    )
    assert isinstance(model, EnsembleCompiledDynamics)
    with TorchDiffEq():
        simulation = model(torch.as_tensor(start_time), torch.as_tensor(end_time))
    check_is_state(simulation, torch.Tensor)
