import pytest
import torch
import pandas as pd
import numpy as np

from pyciemss.compiled_dynamics import CompiledDynamics
from pyciemss.interfaces import sample

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
def test_sample_no_interventions(
    url, start_time, end_time, logging_step_size, num_samples
):
    result = sample(
        url, end_time, logging_step_size, num_samples, start_time=start_time
    )["unprocessed_result"]
    assert isinstance(result, dict)
    check_result_sizes(result, start_time, end_time, logging_step_size, num_samples)


@pytest.mark.parametrize("url", MODEL_URLS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("logging_step_size", LOGGING_STEP_SIZES)
@pytest.mark.parametrize("num_samples", NUM_SAMPLES)
def test_sample_with_static_interventions(
    url, start_time, end_time, logging_step_size, num_samples
):
    model = CompiledDynamics.load(url)

    initial_state = model.initial_state()
    intervened_state_1 = {k: v + 1 for k, v in initial_state.items()}
    intervened_state_2 = {k: v + 2 for k, v in initial_state.items()}

    intervention_time_1 = (end_time + start_time) / 2  # Midpoint
    intervention_time_2 = (end_time + intervention_time_1) / 2  # 3/4 point
    static_interventions = {
        intervention_time_1: intervened_state_1,
        intervention_time_2: intervened_state_2,
    }

    intervened_result = sample(
        url,
        end_time,
        logging_step_size,
        num_samples,
        start_time=start_time,
        static_interventions=static_interventions,
    )["unprocessed_result"]

    result = sample(
        url, end_time, logging_step_size, num_samples, start_time=start_time
    )["unprocessed_result"]

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
def test_sample_with_dynamic_interventions(
    url, start_time, end_time, logging_step_size, num_samples
):
    model = CompiledDynamics.load(url)

    initial_state = model.initial_state()
    intervened_state_1 = {k: v + 1 for k, v in initial_state.items()}
    intervened_state_2 = {k: v + 2 for k, v in initial_state.items()}

    intervention_time_1 = (end_time + start_time) / 2  # Midpoint
    intervention_time_2 = (end_time + intervention_time_1) / 2  # 3/4 point

    def intervention_event_fn_1(time: torch.Tensor, *args, **kwargs):
        return time - intervention_time_1

    def intervention_event_fn_2(time: torch.Tensor, *args, **kwargs):
        return time - intervention_time_2

    dynamic_interventions = {
        intervention_event_fn_1: intervened_state_1,
        intervention_event_fn_2: intervened_state_2,
    }

    intervened_result = sample(
        url,
        end_time,
        logging_step_size,
        num_samples,
        start_time=start_time,
        dynamic_interventions=dynamic_interventions,
    )["unprocessed_result"]

    result = sample(
        url, end_time, logging_step_size, num_samples, start_time=start_time
    )["unprocessed_result"]

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
def test_sample_with_static_and_dynamic_interventions(
    url, start_time, end_time, logging_step_size, num_samples
):
    model = CompiledDynamics.load(url)

    initial_state = model.initial_state()
    intervened_state_1 = {k: v + 1 for k, v in initial_state.items()}
    intervened_state_2 = {k: v + 2 for k, v in initial_state.items()}

    intervention_time_1 = (end_time + start_time) / 2  # Midpoint
    intervention_time_2 = (end_time + intervention_time_1) / 2  # 3/4 point

    def intervention_event_fn_1(time: torch.Tensor, *args, **kwargs):
        return time - intervention_time_1

    dynamic_interventions = {intervention_event_fn_1: intervened_state_1}

    static_interventions = {intervention_time_2: intervened_state_2}

    intervened_result = sample(
        url,
        end_time,
        logging_step_size,
        num_samples,
        start_time=start_time,
        static_interventions=static_interventions,
        dynamic_interventions=dynamic_interventions,
    )["unprocessed_result"]

    result = sample(
        url, end_time, logging_step_size, num_samples, start_time=start_time
    )["unprocessed_result"]

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
def test_output_format(url, start_time, end_time, logging_step_size, num_samples):
    processed_result = sample(
        url, end_time, logging_step_size, num_samples, start_time=start_time
    )["data"]
    assert isinstance(processed_result, pd.DataFrame)
    assert processed_result.shape[0] == num_samples * len(torch.arange(start_time+logging_step_size, end_time, logging_step_size))
    assert processed_result.shape[1] >= 2
    assert list(processed_result.columns)[:2] == ["timepoint_id", "sample_id"]
    for col_name in processed_result.columns[2:]:
        assert col_name.split("_")[-1] in ("param", "state", "(unknown)")
        assert processed_result[col_name].dtype == np.float64

    assert processed_result["timepoint_id"].dtype == np.int64
    assert processed_result["sample_id"].dtype == np.int64