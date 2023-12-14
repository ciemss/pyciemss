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
    MODELS,
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


def setup_calibrate(model_url, start_time, end_time, logging_step_size):
    sample_args = [model_url, end_time, logging_step_size, 1]

    sample_kwargs = {
        "start_time": start_time,
    }

    result = sample(*sample_args, **sample_kwargs)["unprocessed_result"]

    data = {
        k[:-6]: v.squeeze().detach() for k, v in result.items() if k[-5:] == "state"
    }

    data_timespan = torch.arange(
        start_time + logging_step_size, end_time, logging_step_size
    )

    parameter_names = [k for k, v in result.items() if v.ndim == 1]

    return data, data_timespan, parameter_names, sample_args, sample_kwargs


SAMPLE_METHODS = [sample, dummy_ensemble_sample]
INTERVENTION_TYPES = ["static", "dynamic"]
INTERVENTION_TARGETS = ["state", "parameter"]

CALIBRATE_KWARGS = {
    "noise_model": "normal",
    "noise_model_kwargs": {"scale": 1.0},
    "num_iterations": 2,
}


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
            noisy = result[f"{k[:-6]}_noisy"]
            state = result[k]
            assert 0.1 * scale < torch.std(noisy / state - 1) < 10 * scale


@pytest.mark.parametrize("model_fixture", MODELS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("logging_step_size", LOGGING_STEP_SIZES)
@pytest.mark.parametrize("num_samples", NUM_SAMPLES)
@pytest.mark.parametrize("intervention_type", INTERVENTION_TYPES)
@pytest.mark.parametrize("intervention_target", INTERVENTION_TARGETS)
def test_sample_with_interventions(
    model_fixture,
    start_time,
    end_time,
    logging_step_size,
    num_samples,
    intervention_type,
    intervention_target,
):
    model_url = model_fixture.url
    model = CompiledDynamics.load(model_url)

    initial_state = model.initial_state()
    important_parameter_name = model_fixture.important_parameter
    important_parameter = getattr(model, important_parameter_name)

    intervention_effect = 1.0
    intervention_time = (end_time + start_time) / 4  # Quarter of the way through

    if intervention_target == "state":
        intervention = {
            k: v.detach() + intervention_effect for k, v in initial_state.items()
        }
    elif intervention_target == "parameter":
        intervention = {
            important_parameter_name: important_parameter.detach() + intervention_effect
        }

    if intervention_type == "static":
        time_key = intervention_time
    elif intervention_type == "dynamic":
        # Same intervention time expressed as an event function
        def time_key(time, _):
            return time - intervention_time

    interventions_kwargs = {
        f"{intervention_type}_{intervention_target}_interventions": {
            time_key: intervention
        }
    }

    model_args = [model_url, end_time, logging_step_size, num_samples]
    model_kwargs = {"start_time": start_time}

    with pyro.poutine.seed(rng_seed=0):
        intervened_result = sample(
            *model_args,
            **model_kwargs,
            **interventions_kwargs,
        )["unprocessed_result"]

    with pyro.poutine.seed(rng_seed=0):
        result = sample(*model_args, **model_kwargs)["unprocessed_result"]

    check_states_match_in_all_but_values(result, intervened_result)
    check_result_sizes(result, start_time, end_time, logging_step_size, num_samples)
    check_result_sizes(
        intervened_result, start_time, end_time, logging_step_size, num_samples
    )


@pytest.mark.parametrize("model_url", MODEL_URLS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("logging_step_size", LOGGING_STEP_SIZES)
def test_calibrate_no_kwargs(model_url, start_time, end_time, logging_step_size):
    data, data_timespan, _, sample_args, sample_kwargs = setup_calibrate(
        model_url, start_time, end_time, logging_step_size
    )

    calibrate_args = [model_url, data, data_timespan]

    calibrate_kwargs = {"start_time": start_time, **CALIBRATE_KWARGS}

    with pyro.poutine.seed(rng_seed=0):
        inferred_parameters, _ = calibrate(*calibrate_args, **calibrate_kwargs)

    assert isinstance(inferred_parameters, pyro.nn.PyroModule)

    with pyro.poutine.seed(rng_seed=0):
        param_sample_1 = inferred_parameters()

    with pyro.poutine.seed(rng_seed=1):
        param_sample_2 = inferred_parameters()

    for param_name, param_value in param_sample_1.items():
        assert not torch.allclose(param_value, param_sample_2[param_name])

    result = sample(
        *sample_args, **sample_kwargs, inferred_parameters=inferred_parameters
    )["unprocessed_result"]

    check_result_sizes(result, start_time, end_time, logging_step_size, 1)


@pytest.mark.parametrize("model_url", MODEL_URLS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("logging_step_size", LOGGING_STEP_SIZES)
def test_calibrate_deterministic(model_url, start_time, end_time, logging_step_size):
    (
        data,
        data_timespan,
        deterministic_learnable_parameters,
        sample_args,
        sample_kwargs,
    ) = setup_calibrate(model_url, start_time, end_time, logging_step_size)
    calibrate_args = [model_url, data, data_timespan]

    calibrate_kwargs = {
        "start_time": start_time,
        "deterministic_learnable_parameters": deterministic_learnable_parameters
        ** CALIBRATE_KWARGS,
    }

    with pyro.poutine.seed(rng_seed=0):
        inferred_parameters, _ = calibrate(*calibrate_args, **calibrate_kwargs)

    assert isinstance(inferred_parameters, pyro.nn.PyroModule)

    with pyro.poutine.seed(rng_seed=0):
        param_sample_1 = inferred_parameters()

    with pyro.poutine.seed(rng_seed=1):
        param_sample_2 = inferred_parameters()

    for param_name, param_value in param_sample_1.items():
        assert torch.allclose(param_value, param_sample_2[param_name])

    result = sample(
        *sample_args, **sample_kwargs, inferred_parameters=inferred_parameters
    )["unprocessed_result"]

    check_result_sizes(result, start_time, end_time, logging_step_size, 1)


@pytest.mark.parametrize("model_fixture", MODELS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("logging_step_size", LOGGING_STEP_SIZES)
@pytest.mark.parametrize("intervention_type", INTERVENTION_TYPES)
@pytest.mark.parametrize("intervention_target", INTERVENTION_TARGETS)
def test_calibrate_interventions(
    model_fixture,
    start_time,
    end_time,
    logging_step_size,
    intervention_type,
    intervention_target,
):
    model_url = model_fixture.url

    (
        data,
        data_timespan,
        deterministic_learnable_parameters,
        sample_args,
        sample_kwargs,
    ) = setup_calibrate(model_url, start_time, end_time, logging_step_size)
    calibrate_args = [model_url, data, data_timespan]

    calibrate_kwargs = {
        "start_time": start_time,
        "deterministic_learnable_parameters": deterministic_learnable_parameters,
        **CALIBRATE_KWARGS,
    }

    with pyro.poutine.seed(rng_seed=0):
        _, loss = calibrate(*calibrate_args, **calibrate_kwargs)

    # SETUP INTERVENTION

    model = CompiledDynamics.load(model_url)

    intervention_time = (end_time + start_time) / 2  # Half way through

    if intervention_target == "state":
        initial_state = model.initial_state()
        intervention = {k: v.detach() + 1 for k, v in initial_state.items()}
    elif intervention_target == "parameter":
        important_parameter_name = model_fixture.important_parameter
        important_parameter = getattr(model, important_parameter_name)
        intervention = {important_parameter_name: important_parameter.detach() + 1}

    if intervention_type == "static":
        time_key = intervention_time
    elif intervention_type == "dynamic":
        # Same intervention time expressed as an event function
        def time_key(time, _):
            return time - intervention_time

    calibrate_kwargs[f"{intervention_type}_{intervention_target}_interventions"] = {
        time_key: intervention
    }

    with pyro.poutine.seed(rng_seed=0):
        intervened_parameters, intervened_loss = calibrate(
            *calibrate_args, **calibrate_kwargs
        )

    assert intervened_loss != loss

    result = sample(
        *sample_args, **sample_kwargs, inferred_parameters=intervened_parameters
    )["unprocessed_result"]

    check_result_sizes(result, start_time, end_time, logging_step_size, 1)


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
        assert col_name.split("_")[-1] in ("param", "state")
        assert processed_result[col_name].dtype == np.float64

    assert processed_result["timepoint_id"].dtype == np.int64
    assert processed_result["sample_id"].dtype == np.int64
