from copy import deepcopy

import numpy as np
import pandas as pd
import pyro
import pytest
import torch

from pyciemss.compiled_dynamics import CompiledDynamics
from pyciemss.integration_utils.intervention_builder import (
    combine_static_parameter_interventions,
)
from pyciemss.integration_utils.observation import load_data
from pyciemss.interfaces import (
    calibrate,
    ensemble_calibrate,
    ensemble_sample,
    optimize,
    sample,
)

from .fixtures import (
    BAD_AMRS,
    BADLY_FORMATTED_DATAFRAMES,
    END_TIMES,
    LOGGING_STEP_SIZES,
    MAPPING_FOR_DATA_TESTS,
    MODEL_URLS,
    MODELS,
    NON_POS_INTS,
    NUM_SAMPLES,
    OPT_MODELS,
    SEIRHD_NPI_STATIC_PARAM_INTERV,
    START_TIMES,
    check_result_sizes,
    check_states_match,
    check_states_match_in_all_but_values,
)


def dummy_ensemble_sample(model_path_or_json, *args, **kwargs):
    model_paths_or_jsons = [model_path_or_json, model_path_or_json]
    solution_mappings = [
        lambda x: {"total": sum([v for v in x.values()])},
        lambda x: {"total": sum([v for v in x.values()]) / 2},
    ]
    return ensemble_sample(model_paths_or_jsons, solution_mappings, *args, **kwargs)


def dummy_ensemble_calibrate(model_path_or_json, *args, **kwargs):
    model_paths_or_jsons = [model_path_or_json, model_path_or_json]
    solution_mappings = [
        lambda x: x,
        lambda x: {k: v / 2 for k, v in x.items()},
    ]
    return ensemble_calibrate(model_paths_or_jsons, solution_mappings, *args, **kwargs)


def setup_calibrate(model_fixture, start_time, end_time, logging_step_size):
    if model_fixture.data_path is None:
        pytest.skip("TODO: create temporary file")

    data_timepoints = load_data(model_fixture.data_path)[0]

    calibrate_end_time = data_timepoints[-1]

    model_url = model_fixture.url

    sample_args = [model_url, end_time, logging_step_size, 1]
    sample_kwargs = {"start_time": start_time}

    result = sample(*sample_args, **sample_kwargs)["unprocessed_result"]

    parameter_names = [k for k, v in result.items() if v.ndim == 1]

    return parameter_names, calibrate_end_time, sample_args, sample_kwargs


SAMPLE_METHODS = [sample, dummy_ensemble_sample]
CALIBRATE_METHODS = [calibrate, dummy_ensemble_calibrate]
INTERVENTION_TYPES = ["static", "dynamic"]
INTERVENTION_TARGETS = ["state", "parameter"]

CALIBRATE_KWARGS = {
    "noise_model": "normal",
    "noise_model_kwargs": {"scale": 1.0},
    "num_iterations": 2,
}

RTOL = [1e-6, 1e-4]
ATOL = [1e-8, 1e-6]


@pytest.mark.parametrize("sample_method", SAMPLE_METHODS)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("logging_step_size", LOGGING_STEP_SIZES)
@pytest.mark.parametrize("num_samples", NUM_SAMPLES)
@pytest.mark.parametrize("rtol", RTOL)
@pytest.mark.parametrize("atol", ATOL)
def test_sample_no_interventions(
    sample_method,
    model,
    start_time,
    end_time,
    logging_step_size,
    num_samples,
    rtol,
    atol,
):
    model_url = model.url

    with pyro.poutine.seed(rng_seed=0):
        result1 = sample_method(
            model_url,
            end_time,
            logging_step_size,
            num_samples,
            start_time=start_time,
            solver_options={"rtol": rtol, "atol": atol},
        )["unprocessed_result"]
    with pyro.poutine.seed(rng_seed=0):
        result2 = sample_method(
            model_url,
            end_time,
            logging_step_size,
            num_samples,
            start_time=start_time,
            solver_options={"rtol": rtol, "atol": atol},
        )["unprocessed_result"]

    result3 = sample_method(
        model_url,
        end_time,
        logging_step_size,
        num_samples,
        start_time=start_time,
        solver_options={"rtol": rtol, "atol": atol},
    )["unprocessed_result"]

    for result in [result1, result2, result3]:
        assert isinstance(result, dict)
        check_result_sizes(result, start_time, end_time, logging_step_size, num_samples)

    check_states_match(result1, result2)
    if model.has_distributional_parameters:
        check_states_match_in_all_but_values(result1, result3)

    if sample_method.__name__ == "dummy_ensemble_sample":
        assert "total_state" in result1.keys()


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

    def check_noise(noisy, state, scale):
        0.1 * scale < torch.std(noisy / state - 1) < 10 * scale

    if sample_method.__name__ == "dummy_ensemble_sample":
        assert "total_noisy" in result.keys()
        check_noise(result["total_noisy"], result["total_state"], scale)
    else:
        for k, v in result.items():
            if v.ndim == 2 and k[-6:] != "_noisy":
                state_str = k[: k.rfind("_")]
                noisy = result[f"{state_str}_noisy"]

                check_noise(noisy, v, scale)


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

    intervened_result_subset = {
        k: v
        for k, v in intervened_result.items()
        if not k.startswith("parameter_intervention_")
    }
    check_states_match_in_all_but_values(result, intervened_result_subset)

    check_result_sizes(result, start_time, end_time, logging_step_size, num_samples)
    check_result_sizes(
        intervened_result, start_time, end_time, logging_step_size, num_samples
    )


@pytest.mark.parametrize("model_fixture", MODELS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("logging_step_size", LOGGING_STEP_SIZES)
@pytest.mark.parametrize("num_samples", NUM_SAMPLES)
def test_sample_with_multiple_parameter_interventions(
    model_fixture,
    start_time,
    end_time,
    logging_step_size,
    num_samples,
):
    model_url = model_fixture.url
    model = CompiledDynamics.load(model_url)

    important_parameter_name = model_fixture.important_parameter
    important_parameter = getattr(model, important_parameter_name)

    intervention_effect = 1.1
    intervention_time_0 = (end_time + start_time) / 4  # Quarter of the way through
    intervention_time_1 = (end_time + start_time) / 2  # Half way through

    intervention_0 = {
        important_parameter_name: important_parameter.detach() * intervention_effect
    }

    intervention_1 = {
        important_parameter_name: important_parameter.detach() / intervention_effect
    }

    model_args = [model_url, end_time, logging_step_size, num_samples]
    model_kwargs = {"start_time": start_time}

    with pyro.poutine.seed(rng_seed=0):
        intervened_result = sample(
            *model_args,
            **model_kwargs,
            static_parameter_interventions={
                intervention_time_0: intervention_0,
                intervention_time_1: intervention_1,
            },
        )["unprocessed_result"]

    assert "parameter_intervention_time_0" in intervened_result.keys()
    assert (
        torch.isclose(
            intervened_result["parameter_intervention_time_0"],
            torch.as_tensor(intervention_time_0),
        )
        .all()
        .item()
    )
    assert (
        f"parameter_intervention_value_{important_parameter_name}_0"
        in intervened_result.keys()
    )
    assert "parameter_intervention_time_1" in intervened_result.keys()
    assert (
        torch.isclose(
            intervened_result["parameter_intervention_time_1"],
            torch.as_tensor(intervention_time_1),
        )
        .all()
        .item()
    )
    assert (
        f"parameter_intervention_value_{important_parameter_name}_1"
        in intervened_result.keys()
    )

    check_result_sizes(
        intervened_result, start_time, end_time, logging_step_size, num_samples
    )


@pytest.mark.parametrize("calibrate_method", CALIBRATE_METHODS)
@pytest.mark.parametrize("model_fixture", MODELS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("logging_step_size", LOGGING_STEP_SIZES)
def test_calibrate_no_kwargs(
    calibrate_method, model_fixture, start_time, end_time, logging_step_size
):
    model_url = model_fixture.url
    _, _, sample_args, sample_kwargs = setup_calibrate(
        model_fixture, start_time, end_time, logging_step_size
    )

    calibrate_args = [model_url, model_fixture.data_path]

    calibrate_kwargs = {
        "data_mapping": model_fixture.data_mapping,
        "start_time": start_time,
        **CALIBRATE_KWARGS,
    }

    with pyro.poutine.seed(rng_seed=0):
        inferred_parameters = calibrate_method(*calibrate_args, **calibrate_kwargs)[
            "inferred_parameters"
        ]

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


@pytest.mark.parametrize("model_fixture", MODELS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("logging_step_size", LOGGING_STEP_SIZES)
@pytest.mark.parametrize("rtol", RTOL)
@pytest.mark.parametrize("atol", ATOL)
def test_calibrate_deterministic(
    model_fixture, start_time, end_time, logging_step_size, rtol, atol
):
    model_url = model_fixture.url
    (
        deterministic_learnable_parameters,
        _,
        sample_args,
        sample_kwargs,
    ) = setup_calibrate(model_fixture, start_time, end_time, logging_step_size)

    calibrate_args = [model_url, model_fixture.data_path]

    calibrate_kwargs = {
        "data_mapping": model_fixture.data_mapping,
        "start_time": start_time,
        "deterministic_learnable_parameters": deterministic_learnable_parameters,
        "solver_options": {"rtol": rtol, "atol": atol},
        **CALIBRATE_KWARGS,
    }

    with pyro.poutine.seed(rng_seed=0):
        output = calibrate(*calibrate_args, **calibrate_kwargs)
        inferred_parameters = output["inferred_parameters"]

    assert isinstance(inferred_parameters, pyro.nn.PyroModule)

    with pyro.poutine.seed(rng_seed=0):
        param_sample_1 = inferred_parameters()

    with pyro.poutine.seed(rng_seed=1):
        param_sample_2 = inferred_parameters()

    for param_name, param_value in param_sample_1.items():
        assert torch.allclose(param_value, param_sample_2[param_name])

    result = sample(
        *sample_args,
        **sample_kwargs,
        inferred_parameters=inferred_parameters,
        solver_options={"rtol": rtol, "atol": atol},
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
        deterministic_learnable_parameters,
        calibrate_end_time,
        sample_args,
        sample_kwargs,
    ) = setup_calibrate(model_fixture, start_time, end_time, logging_step_size)
    calibrate_args = [model_url, model_fixture.data_path]

    calibrate_kwargs = {
        "data_mapping": model_fixture.data_mapping,
        "start_time": start_time,
        "deterministic_learnable_parameters": deterministic_learnable_parameters,
        **CALIBRATE_KWARGS,
    }

    with pyro.poutine.seed(rng_seed=0):
        loss = calibrate(*calibrate_args, **calibrate_kwargs)["loss"]

    # SETUP INTERVENTION

    model = CompiledDynamics.load(model_url)

    intervention_time = (calibrate_end_time + start_time) / 2  # Half way through

    if intervention_target == "state":
        initial_state = model.initial_state()
        intervention = {k: (lambda x: 0.9 * x) for k in initial_state.keys()}
    elif intervention_target == "parameter":
        important_parameter_name = model_fixture.important_parameter
        intervention = {important_parameter_name: (lambda x: 0.9 * x)}

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
        output = calibrate(*calibrate_args, **calibrate_kwargs)

        intervened_parameters, intervened_loss = (
            output["inferred_parameters"],
            output["loss"],
        )

    assert intervened_loss != loss

    result = sample(
        *sample_args, **sample_kwargs, inferred_parameters=intervened_parameters
    )["unprocessed_result"]

    check_result_sizes(result, start_time, end_time, logging_step_size, 1)


@pytest.mark.parametrize("model_fixture", MODELS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("logging_step_size", LOGGING_STEP_SIZES)
def test_calibrate_progress_hook(
    model_fixture, start_time, end_time, logging_step_size
):
    model_url = model_fixture.url

    (
        _,
        calibrate_end_time,
        sample_args,
        sample_kwargs,
    ) = setup_calibrate(model_fixture, start_time, end_time, logging_step_size)

    calibrate_args = [model_url, model_fixture.data_path]

    class TestProgressHook:
        def __init__(self):
            self.iterations = []
            self.losses = []

        def __call__(self, iteration, loss):
            # Log the loss and iteration number
            self.iterations.append(iteration)
            self.losses.append(loss)

    progress_hook = TestProgressHook()

    calibrate_kwargs = {
        "data_mapping": model_fixture.data_mapping,
        "start_time": start_time,
        "progress_hook": progress_hook,
        **CALIBRATE_KWARGS,
    }

    calibrate(*calibrate_args, **calibrate_kwargs)

    assert len(progress_hook.iterations) == CALIBRATE_KWARGS["num_iterations"]
    assert len(progress_hook.losses) == CALIBRATE_KWARGS["num_iterations"]


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
        url,
        end_time,
        logging_step_size,
        num_samples,
        start_time=start_time,
        time_unit="nominal",
    )["data"]
    assert isinstance(processed_result, pd.DataFrame)
    assert processed_result.shape[0] == num_samples * len(
        torch.arange(start_time, end_time + logging_step_size, logging_step_size)
    )
    assert processed_result.shape[1] >= 2
    assert list(processed_result.columns)[:3] == [
        "timepoint_id",
        "sample_id",
        "timepoint_nominal",
    ]
    for col_name in processed_result.columns[3:]:
        assert col_name.split("_")[-1] in ("param", "state")
        assert processed_result[col_name].dtype == np.float64

    assert processed_result["timepoint_id"].dtype == np.int64
    assert processed_result["sample_id"].dtype == np.int64


@pytest.mark.parametrize("model_fixture", OPT_MODELS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("num_samples", NUM_SAMPLES)
@pytest.mark.parametrize("rtol", RTOL)
@pytest.mark.parametrize("atol", ATOL)
def test_optimize(model_fixture, start_time, end_time, num_samples, rtol, atol):
    logging_step_size = 1.0
    model_url = model_fixture.url

    class TestProgressHook:
        def __init__(self):
            self.result_x = []

        def __call__(self, x):
            # Log the iteration number
            self.result_x.append(x)
            print(f"Result: {self.result_x}")

    progress_hook = TestProgressHook()

    optimize_kwargs = {
        **model_fixture.optimize_kwargs,
        "solver_method": "euler",
        "solver_options": {"step_size": 1.0, "rtol": rtol, "atol": atol},
        "start_time": start_time,
        "n_samples_ouu": int(2),
        "maxiter": 1,
        "maxfeval": 2,
        "progress_hook": progress_hook,
    }
    bounds_interventions = optimize_kwargs["bounds_interventions"]
    opt_result = optimize(
        model_url,
        end_time,
        logging_step_size,
        **optimize_kwargs,
    )
    opt_policy = opt_result["policy"]
    for i in range(opt_policy.shape[-1]):
        assert bounds_interventions[0][i] <= opt_policy[i]
        assert opt_policy[i] <= bounds_interventions[1][i]

    opt_intervention_temp = optimize_kwargs["static_parameter_interventions"](
        opt_result["policy"]
    )
    if "fixed_static_parameter_interventions" in optimize_kwargs:
        intervention_list = [
            deepcopy(optimize_kwargs["fixed_static_parameter_interventions"])
        ]
        intervention_list.extend(
            [opt_intervention_temp]
            if not isinstance(opt_intervention_temp, list)
            else opt_intervention_temp
        )
        opt_intervention = combine_static_parameter_interventions(intervention_list)
    else:
        opt_intervention = opt_intervention_temp
    if "fixed_static_state_interventions" not in optimize_kwargs:
        fixed_static_state_interventions = {}
    else:
        fixed_static_state_interventions = optimize_kwargs[
            "fixed_static_state_interventions"
        ]
    if "fixed_dynamic_parameter_interventions" not in optimize_kwargs:
        fixed_dynamic_parameter_interventions = {}
    else:
        fixed_dynamic_parameter_interventions = optimize_kwargs[
            "fixed_dynamic_parameter_interventions"
        ]
    if "fixed_dynamic_state_interventions" not in optimize_kwargs:
        fixed_dynamic_state_interventions = {}
    else:
        fixed_dynamic_state_interventions = optimize_kwargs[
            "fixed_dynamic_state_interventions"
        ]

    if "alpha" in optimize_kwargs:
        alpha = optimize_kwargs["alpha"]
    else:
        alpha = [0.95]
    with pyro.poutine.seed(rng_seed=0):
        result_opt = sample(
            model_url,
            end_time,
            logging_step_size,
            num_samples,
            start_time=start_time,
            static_parameter_interventions=opt_intervention,
            static_state_interventions=fixed_static_state_interventions,
            dynamic_parameter_interventions=fixed_dynamic_parameter_interventions,
            dynamic_state_interventions=fixed_dynamic_state_interventions,
            solver_method=optimize_kwargs["solver_method"],
            solver_options=optimize_kwargs["solver_options"],
            alpha=alpha,
            qoi=optimize_kwargs["qoi"],
        )["unprocessed_result"]

    intervened_result_subset = {
        k: v
        for k, v in result_opt.items()
        if not k.startswith("parameter_intervention_")
    }

    assert isinstance(intervened_result_subset, dict)
    check_result_sizes(
        intervened_result_subset, start_time, end_time, logging_step_size, num_samples
    )

    assert len(progress_hook.result_x) <= (
        (optimize_kwargs["maxfeval"] + 1) * (optimize_kwargs["maxiter"] + 1)
    )


@pytest.mark.parametrize("model_fixture", MODELS)
@pytest.mark.parametrize("bad_num_iterations", NON_POS_INTS)
def test_non_pos_int_calibrate(model_fixture, bad_num_iterations):
    # Assert that a ValueError is raised when num_iterations is not a positive integer
    if model_fixture.data_path is None or model_fixture.data_mapping is None:
        pytest.skip("Skip models with no data attached")
    with pytest.raises(ValueError):
        calibrate(
            model_fixture.url,
            model_fixture.data_path,
            data_mapping=model_fixture.data_mapping,
            num_iterations=bad_num_iterations,
        )


@pytest.mark.parametrize("sample_method", SAMPLE_METHODS)
@pytest.mark.parametrize("model_url", MODEL_URLS)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("logging_step_size", LOGGING_STEP_SIZES)
@pytest.mark.parametrize("bad_num_samples", NON_POS_INTS)
def test_non_pos_int_sample(
    sample_method, model_url, end_time, logging_step_size, bad_num_samples
):
    # Assert that a ValueError is raised when num_samples is not a positive integer
    with pytest.raises(ValueError):
        sample_method(
            model_url,
            end_time,
            logging_step_size,
            num_samples=bad_num_samples,
        )


@pytest.mark.parametrize("bad_data", BADLY_FORMATTED_DATAFRAMES)
@pytest.mark.parametrize("data_mapping", MAPPING_FOR_DATA_TESTS)
def test_load_data(bad_data, data_mapping):
    # Assert that a ValueError is raised for improperly formatted data
    with pytest.raises(ValueError):
        load_data(
            bad_data,
            data_mapping,
        )


@pytest.mark.parametrize("model_fixture", MODELS)
def test_bad_euler_solver_calibrate(model_fixture):
    # Assert that a ValueError is raised when the 'step_size' option is not provided for the 'euler' solver method
    if model_fixture.data_path is None or model_fixture.data_mapping is None:
        pytest.skip("Skip models with no data attached")
    with pytest.raises(ValueError):
        calibrate(
            model_fixture.url,
            model_fixture.data_path,
            data_mapping=model_fixture.data_mapping,
            solver_method="euler",
            solver_options={},
        )


@pytest.mark.parametrize("model_fixture", MODELS)
@pytest.mark.parametrize("sample_method", SAMPLE_METHODS)
def test_bad_euler_solver_sample(model_fixture, sample_method):
    # Assert that a ValueError is raised when the 'step_size' option is not provided for the 'euler' solver method
    with pytest.raises(ValueError):
        sample_method(
            model_fixture.url,
            1,
            1,
            1,
            solver_method="euler",
            solver_options={},
        )


@pytest.mark.parametrize("model_fixture", OPT_MODELS)
def test_bad_euler_solver_optimize(model_fixture):
    # Assert that a ValueError is raised when the 'step_size' option is not provided for the 'euler' solver method
    with pytest.raises(ValueError):
        logging_step_size = 1.0
        model_url = model_fixture.url
        optimize_kwargs = {
            **model_fixture.optimize_kwargs,
            "solver_method": "euler",
            "start_time": 1.0,
            "n_samples_ouu": int(2),
            "maxiter": 1,
            "maxfeval": 2,
            "solver_options": {},
        }
        optimize(
            model_url,
            2.0,
            logging_step_size,
            **optimize_kwargs,
        )


@pytest.mark.parametrize("sample_method", SAMPLE_METHODS)
@pytest.mark.parametrize("model_fixture", BAD_AMRS)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("logging_step_size", LOGGING_STEP_SIZES)
@pytest.mark.parametrize("num_samples", NUM_SAMPLES)
def test_errors_for_bad_amrs(
    sample_method, model_fixture, end_time, logging_step_size, num_samples
):
    # Assert that a ValueError is raised when AMR contains undefined variables
    with pytest.raises(ValueError):
        sample_method(
            model_fixture.url,
            end_time,
            logging_step_size,
            num_samples,
        )


@pytest.mark.parametrize("sample_method", [sample])
@pytest.mark.parametrize("model_fixture", MODELS)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("logging_step_size", LOGGING_STEP_SIZES)
@pytest.mark.parametrize("num_samples", NUM_SAMPLES)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("seirhd_npi_intervention", SEIRHD_NPI_STATIC_PARAM_INTERV)
def test_intervention_on_constant_param(
    sample_method,
    model_fixture,
    end_time,
    logging_step_size,
    num_samples,
    start_time,
    seirhd_npi_intervention,
):
    # Assert that sample returns expected result with intervention on constant parameter
    if "SEIRHD_NPI" not in model_fixture.url:
        pytest.skip("Only test 'SEIRHD_NPI' models with constant parameter delta")
    else:
        processed_result = sample_method(
            model_fixture.url,
            end_time,
            logging_step_size,
            num_samples,
            start_time=start_time,
            static_parameter_interventions=seirhd_npi_intervention,
        )["data"]
        assert isinstance(processed_result, pd.DataFrame)
        assert processed_result.shape[0] == num_samples * len(
            torch.arange(start_time, end_time + logging_step_size, logging_step_size)
        )
        assert processed_result.shape[1] >= 2


@pytest.mark.parametrize("sample_method", [sample])
@pytest.mark.parametrize("model_fixture", MODELS)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("logging_step_size", LOGGING_STEP_SIZES)
@pytest.mark.parametrize("num_samples", NUM_SAMPLES)
@pytest.mark.parametrize("start_time", START_TIMES)
def test_observables_change_with_interventions(
    sample_method,
    model_fixture,
    end_time,
    logging_step_size,
    num_samples,
    start_time,
):
    # Assert that sample returns expected result with intervention on constant parameter
    if "SIR_param" not in model_fixture.url:
        pytest.skip("Only test 'SIR_param_in_obs' model")
    else:
        processed_result = sample_method(
            model_fixture.url,
            end_time,
            logging_step_size,
            num_samples,
            start_time=start_time,
            static_parameter_interventions={
                torch.tensor(2.0): {"beta": torch.tensor(0.001)}
            },
        )["data"]

        # The test will fail if values before and after the intervention are the same
        assert (
            processed_result["beta_param_observable_state"][0]
            > processed_result["beta_param_observable_state"][
                int(end_time / logging_step_size)
            ]
        )
