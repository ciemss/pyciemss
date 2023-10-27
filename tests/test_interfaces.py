import pytest
from chirho.dynamical.ops import State

from pyciemss.compiled_dynamics import CompiledDynamics
from pyciemss.interfaces import simulate

from .model_fixtures import END_TIMES, LOGGING_STEP_SIZES, MODEL_URLS, START_TIMES


@pytest.mark.parametrize("url", MODEL_URLS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("logging_step_size", LOGGING_STEP_SIZES)
def test_simulate(url, start_time, end_time, logging_step_size):
    result = simulate(url, start_time, end_time, logging_step_size)
    assert isinstance(result, State)

    # TODO: test something about the state itself

@pytest.mark.parametrize("url", MODEL_URLS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("logging_step_size", LOGGING_STEP_SIZES)
def test_simulate_with_static_interventions(url, start_time, end_time, logging_step_size):
    model = CompiledDynamics.load(url)

    initial_state = model.initial_state()
    intervention_time_1 = (end_time + start_time)/2 # Midpoint
    intervention_time_2 = (end_time + intervention_time_1)/2 # 3/4 point
    static_interventions = {intervention_time_1: initial_state, intervention_time_2: initial_state}
    
    result = simulate(url, start_time, end_time, logging_step_size, static_interventions=static_interventions)
    assert isinstance(result, State)

    # TODO: test something about the state itself


    
