import pytest
import torch

from pyciemss.compiled_dynamics import CompiledDynamics
from pyciemss.interfaces import simulate

from .fixtures import (
    END_TIMES,
    LOGGING_STEP_SIZES,
    MODEL_URLS,
    NUM_SAMPLES,
    START_TIMES,
    check_result_sizes,
    check_states_match_in_all_but_values,
)


@pytest.mark.parametrize("url", MODEL_URLS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("logging_step_size", LOGGING_STEP_SIZES)
@pytest.mark.parametrize("num_samples", NUM_SAMPLES)
def test_simulate_no_interventions(
    url, start_time, end_time, logging_step_size, num_samples
):
    result = simulate(url, start_time, end_time, logging_step_size, num_samples)
    assert isinstance(result, dict)

    check_result_sizes(result, start_time, end_time, logging_step_size, num_samples)


@pytest.mark.parametrize("url", MODEL_URLS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("logging_step_size", LOGGING_STEP_SIZES)
@pytest.mark.parametrize("num_samples", NUM_SAMPLES)
def test_simulate_with_static_interventions(
    url, start_time, end_time, logging_step_size, num_samples
):
    model = CompiledDynamics.load(url)

    initial_state = model.initial_state()
    intervened_state_1 = {k: v + 1 for k, v in initial_state.items()}
    intervened_state_2 = {k: v + 2 for k, v in initial_state.items()}

    intervention_time_1 = (end_time + start_time) / 2.000001  # Midpoint
    intervention_time_2 = (end_time + intervention_time_1) / 2  # 3/4 point
    static_interventions = {
        intervention_time_1: intervened_state_1,
        intervention_time_2: intervened_state_2,
    }

    intervened_result = simulate(
        url,
        start_time,
        end_time,
        logging_step_size,
        num_samples,
        static_interventions=static_interventions,
    )

    result = simulate(url, start_time, end_time, logging_step_size, num_samples)

    check_states_match_in_all_but_values(result, intervened_result)
    check_result_sizes(result, start_time, end_time, logging_step_size, num_samples)
    check_result_sizes(
        intervened_result, start_time, end_time, logging_step_size, num_samples
    )


@pytest.mark.parametrize("url", MODEL_URLS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("logging_step_size", LOGGING_STEP_SIZES)
@pytest.mark.parametrize("num_samples", NUM_SAMPLES)
def test_simulate_with_dynamic_interventions(
    url, start_time, end_time, logging_step_size, num_samples
):
    model = CompiledDynamics.load(url)

    initial_state = model.initial_state()
    intervened_state_1 = {k: v + 1 for k, v in initial_state.items()}
    intervened_state_2 = {k: v + 2 for k, v in initial_state.items()}

    intervention_time_1 = (end_time + start_time) / 2.000001  # Midpoint
    intervention_time_2 = (end_time + intervention_time_1) / 2  # 3/4 point

    def intervention_event_fn_1(time: torch.Tensor, *args, **kwargs):
        return time - intervention_time_1

    def intervention_event_fn_2(time: torch.Tensor, *args, **kwargs):
        return time - intervention_time_2

    dynamic_interventions = {
        intervention_event_fn_1: intervened_state_1,
        intervention_event_fn_2: intervened_state_2,
    }

    intervened_result = simulate(
        url,
        start_time,
        end_time,
        logging_step_size,
        num_samples,
        dynamic_interventions=dynamic_interventions,
    )

    result = simulate(url, start_time, end_time, logging_step_size, num_samples)

    check_states_match_in_all_but_values(result, intervened_result)
    check_result_sizes(result, start_time, end_time, logging_step_size, num_samples)
    check_result_sizes(
        intervened_result, start_time, end_time, logging_step_size, num_samples
    )