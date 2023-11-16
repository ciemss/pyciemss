import numpy as np
import pandas as pd
import pyro
import pytest
import torch

from pyciemss.compiled_dynamics import CompiledDynamics
from pyciemss.interfaces import calibrate, ensemble_sample, sample

from .fixtures import (
    END_TIMES,
    LOGGING_STEP_SIZES,
    MODEL_URLS,
    NUM_SAMPLES,
    START_TIMES,
    check_result_sizes,
    check_states_match,
    check_states_match_in_all_but_values,
)


def dummy_ensemble_sample(model_path_or_json, *args, **kwargs):
    model_paths_or_jsons = [model_path_or_json, model_path_or_json]
    solution_mappings = [lambda x: x, lambda x: {k: 2 * v for k, v in x.items()}]
    return ensemble_sample(model_paths_or_jsons, solution_mappings, *args, **kwargs)


SAMPLE_METHODS = [sample, dummy_ensemble_sample]


@pytest.mark.parametrize("sample_method", SAMPLE_METHODS)
@pytest.mark.parametrize("model_url", MODEL_URLS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("logging_step_size", LOGGING_STEP_SIZES)
@pytest.mark.parametrize("num_samples", NUM_SAMPLES)
def test_sample_no_interventions(
    sample_method, model_url, start_time, end_time, logging_step_size, num_samples
):
    with pyro.poutine.seed(rng_seed=0):
        result1 = sample_method(
            model_url, end_time, logging_step_size, num_samples, start_time=start_time
        )["unprocessed_result"]
    with pyro.poutine.seed(rng_seed=0):
        result2 = sample_method(
            model_url, end_time, logging_step_size, num_samples, start_time=start_time
        )["unprocessed_result"]

    result3 = sample_method(
        model_url, end_time, logging_step_size, num_samples, start_time=start_time
    )["unprocessed_result"]

    for result in [result1, result2, result3]:
        assert isinstance(result, dict)
        check_result_sizes(result, start_time, end_time, logging_step_size, num_samples)

    check_states_match(result1, result2)
    check_states_match_in_all_but_values(result1, result3)


@pytest.mark.parametrize("sample_method", SAMPLE_METHODS)
@pytest.mark.parametrize("model_url", MODEL_URLS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("logging_step_size", LOGGING_STEP_SIZES)
@pytest.mark.parametrize("num_samples", NUM_SAMPLES)
@pytest.mark.parametrize("scale", [0.1, 10.0])
def test_sample_with_noise(
    sample_method,
    model_url,
    start_time,
    end_time,
    logging_step_size,
    num_samples,
    scale,
):
    result = sample_method(
        model_url,
        end_time,
        logging_step_size,
        num_samples,
        start_time=start_time,
        noise_model="normal",
        noise_model_kwargs={"scale": scale},
    )["unprocessed_result"]
    assert isinstance(result, dict)
    check_result_sizes(result, start_time, end_time, logging_step_size, num_samples)

    for k in result.keys():
        if k[-5:] == "state":
            observed = result[f"{k[:-6]}_observed"]
            state = result[k]
            assert 0.5 * scale < torch.std(observed / state - 1) < 2 * scale


@pytest.mark.parametrize("model_url", MODEL_URLS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("logging_step_size", LOGGING_STEP_SIZES)
@pytest.mark.parametrize("num_samples", NUM_SAMPLES)
def test_sample_with_static_interventions(
    model_url, start_time, end_time, logging_step_size, num_samples
):
    model = CompiledDynamics.load(model_url)

    initial_state = model.initial_state()
    intervened_state_1 = {k: v + 1 for k, v in initial_state.items()}
    intervened_state_2 = {k: v + 2 for k, v in initial_state.items()}

    intervention_time_1 = (end_time + start_time) / 2  # Midpoint
    intervention_time_2 = (end_time + intervention_time_1) / 2  # 3/4 point
    static_interventions = {
        intervention_time_1: intervened_state_1,
        intervention_time_2: intervened_state_2,
    }
    with pyro.poutine.seed(rng_seed=0):
        intervened_result = sample(
            model_url,
            end_time,
            logging_step_size,
            num_samples,
            start_time=start_time,
            static_interventions=static_interventions,
        )["unprocessed_result"]

    with pyro.poutine.seed(rng_seed=0):
        result = sample(
            model_url, end_time, logging_step_size, num_samples, start_time=start_time
        )["unprocessed_result"]

    check_states_match_in_all_but_values(result, intervened_result)
    check_result_sizes(result, start_time, end_time, logging_step_size, num_samples)
    check_result_sizes(
        intervened_result, start_time, end_time, logging_step_size, num_samples
    )


@pytest.mark.parametrize("model_url", MODEL_URLS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("logging_step_size", LOGGING_STEP_SIZES)
@pytest.mark.parametrize("num_samples", NUM_SAMPLES)
def test_sample_with_dynamic_interventions(
    model_url, start_time, end_time, logging_step_size, num_samples
):
    model = CompiledDynamics.load(model_url)

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

    with pyro.poutine.seed(rng_seed=0):
        intervened_result = sample(
            model_url,
            end_time,
            logging_step_size,
            num_samples,
            start_time=start_time,
            dynamic_interventions=dynamic_interventions,
        )["unprocessed_result"]

    with pyro.poutine.seed(rng_seed=0):
        result = sample(
            model_url, end_time, logging_step_size, num_samples, start_time=start_time
        )["unprocessed_result"]

    check_states_match_in_all_but_values(result, intervened_result)
    check_result_sizes(result, start_time, end_time, logging_step_size, num_samples)
    check_result_sizes(
        intervened_result, start_time, end_time, logging_step_size, num_samples
    )


@pytest.mark.parametrize("model_url", MODEL_URLS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("logging_step_size", LOGGING_STEP_SIZES)
@pytest.mark.parametrize("num_samples", NUM_SAMPLES)
def test_sample_with_static_and_dynamic_interventions(
    model_url, start_time, end_time, logging_step_size, num_samples
):
    model = CompiledDynamics.load(model_url)

    initial_state = model.initial_state()
    intervened_state_1 = {k: v + 1 for k, v in initial_state.items()}
    intervened_state_2 = {k: v + 2 for k, v in initial_state.items()}

    intervention_time_1 = (end_time + start_time) / 2  # Midpoint
    intervention_time_2 = (end_time + intervention_time_1) / 2  # 3/4 point

    def intervention_event_fn_1(time: torch.Tensor, *args, **kwargs):
        return time - intervention_time_1

    dynamic_interventions = {intervention_event_fn_1: intervened_state_1}

    static_interventions = {intervention_time_2: intervened_state_2}
    with pyro.poutine.seed(rng_seed=0):
        intervened_result = sample(
            model_url,
            end_time,
            logging_step_size,
            num_samples,
            start_time=start_time,
            static_interventions=static_interventions,
            dynamic_interventions=dynamic_interventions,
        )["unprocessed_result"]
    with pyro.poutine.seed(rng_seed=0):
        result = sample(
            model_url, end_time, logging_step_size, num_samples, start_time=start_time
        )["unprocessed_result"]

    check_states_match_in_all_but_values(result, intervened_result)
    check_result_sizes(result, start_time, end_time, logging_step_size, num_samples)
    check_result_sizes(
        intervened_result, start_time, end_time, logging_step_size, num_samples
    )


@pytest.mark.parametrize("model_url", MODEL_URLS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("logging_step_size", LOGGING_STEP_SIZES)
def test_calibrate_and_sample(model_url, start_time, end_time, logging_step_size):
    result = sample(
        model_url,
        end_time,
        logging_step_size,
        1,
        start_time=start_time,
    )["unprocessed_result"]

    data = {k[:-6]: v.squeeze() for k, v in result.items() if k[-5:] == "state"}
    parameter_names = [k for k in result.keys() if k[-5:] != "state"]

    data_timespan = torch.arange(
        start_time + logging_step_size, end_time, logging_step_size
    )

    inferred_parameters_1 = calibrate(
        model_url,
        data,
        data_timespan,
        start_time=start_time,
        noise_model="normal",
        noise_model_kwargs={"scale": 1.0},
        num_iterations=2,
    )

    inferred_parameters_2 = calibrate(
        model_url,
        data,
        data_timespan,
        start_time=start_time,
        noise_model="normal",
        noise_model_kwargs={"scale": 1.0},
        num_iterations=2,
        deterministic_learnable_parameters=parameter_names,
    )

    assert isinstance(inferred_parameters_1, pyro.nn.PyroModule)
    assert isinstance(inferred_parameters_2, pyro.nn.PyroModule)

    with pyro.poutine.seed(rng_seed=0):
        calibrated_result_1 = sample(
            model_url,
            end_time,
            logging_step_size,
            1,
            start_time=start_time,
            inferred_parameters=inferred_parameters_1,
        )["unprocessed_result"]

    with pyro.poutine.seed(rng_seed=0):
        calibrated_result_2 = sample(
            model_url,
            end_time,
            logging_step_size,
            1,
            start_time=start_time,
            inferred_parameters=inferred_parameters_2,
        )["unprocessed_result"]

    with pyro.poutine.seed(rng_seed=1):
        calibrated_result_3 = sample(
            model_url,
            end_time,
            logging_step_size,
            1,
            start_time=start_time,
            inferred_parameters=inferred_parameters_2,
        )["unprocessed_result"]

    check_states_match_in_all_but_values(calibrated_result_1, result)
    check_states_match_in_all_but_values(calibrated_result_2, result)
    check_states_match(calibrated_result_2, calibrated_result_3)


@pytest.mark.parametrize("sample_method", SAMPLE_METHODS)
@pytest.mark.parametrize("url", MODEL_URLS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("logging_step_size", LOGGING_STEP_SIZES)
@pytest.mark.parametrize("num_samples", NUM_SAMPLES)
def test_output_format(
    sample_method, url, start_time, end_time, logging_step_size, num_samples
):
    processed_result = sample_method(
        url, end_time, logging_step_size, num_samples, start_time=start_time
    )["data"]
    assert isinstance(processed_result, pd.DataFrame)
    assert processed_result.shape[0] == num_samples * len(
        torch.arange(start_time + logging_step_size, end_time, logging_step_size)
    )
    assert processed_result.shape[1] >= 2
    assert list(processed_result.columns)[:2] == ["timepoint_id", "sample_id"]
    for col_name in processed_result.columns[2:]:
        assert col_name.split("_")[-1] in ("param", "state", "(unknown)")
        assert processed_result[col_name].dtype == np.float64

    assert processed_result["timepoint_id"].dtype == np.int64
    assert processed_result["sample_id"].dtype == np.int64
