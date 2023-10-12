import json
import logging

import mira
import pytest
import torch

from chirho.dynamical.handlers import SimulatorEventLoop, StaticIntervention, DynamicTrace
from chirho.dynamical.ops import State, Trajectory
from pyciemss.mira import CompiledInPlaceDynamics, default_initial_state


logger = logging.getLogger(__name__)

start_time = torch.tensor(0.0)
end_time = torch.tensor(4.0)
logging_times = torch.tensor([1.0, 2.0, 3.0])

@pytest.mark.parametrize(
    "model_path",
    [
        "test/models/AMR_examples/BIOMD0000000955_askenet.json",
        "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/sir_typed.json",
    ],
)
def test_simulate_from_askenet(model_path):
    mira_model = CompiledInPlaceDynamics.from_askenet(model_path)

    assert isinstance(mira_model, CompiledInPlaceDynamics)
    assert isinstance(mira_model.src, mira.modeling.Model)

    initial_state = default_initial_state(mira_model.src)
    
    assert isinstance(initial_state, State)
    assert len(initial_state.keys) > 0

    with DynamicTrace(logging_times) as dt:
        result = mira_model(initial_state, start_time, end_time)
    assert isinstance(result, State)
    assert result.keys == initial_state.keys
    
    assert isinstance(dt.trace, Trajectory)
    assert dt.trace.keys == initial_state.keys

    for key in result.keys:
        value = getattr(dt.trace, key)
        assert isinstance(value, torch.Tensor)
        assert value.shape[0] == logging_times.shape[0] > 1
        assert not torch.any(torch.isnan(value))
        assert torch.any(value[1:] != value[0])


@pytest.mark.parametrize(
    "model_path",
    [
        "test/models/AMR_examples/BIOMD0000000955_askenet.json",
        "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/sir_typed.json",
    ],
)
def test_simulate_intervened_from_askenet(model_path):
    mira_model = CompiledInPlaceDynamics.from_askenet(model_path)

    assert isinstance(mira_model, CompiledInPlaceDynamics)
    assert isinstance(mira_model.src, mira.modeling.Model)

    initial_state = default_initial_state(mira_model.src)
    assert isinstance(initial_state, State)
    assert len(initial_state.keys) > 0

    with DynamicTrace(logging_times) as dt, \
            SimulatorEventLoop(), \
            StaticIntervention(time=0.15, intervention=initial_state), \
            StaticIntervention(time=0.3, intervention=initial_state):
        result = mira_model(initial_state, start_time, end_time)

    assert isinstance(result, State)
    assert result.keys == initial_state.keys
    
    assert isinstance(dt.trace, Trajectory)
    assert dt.trace.keys == initial_state.keys

    for key in result.keys:
        value = getattr(dt.trace, key)
        assert isinstance(value, torch.Tensor)
        assert value.shape[0] == logging_times.shape[0] > 1
        assert not torch.any(torch.isnan(value))
        assert torch.any(value[1:] != value[0])