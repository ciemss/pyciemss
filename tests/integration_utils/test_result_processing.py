import pytest
import torch

import pyciemss
from pyciemss.integration_utils import result_processing

intervention_times = {
    "parameter_intervention_time_0": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]),
    "parameter_intervention_time_1": torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
    "parameter_intervention_time_20": torch.tensor([20.0, 20.0, 20.0, 20.0, 20.0]),
}

intervention_values = {
    "parameter_intervention_value_p_cbeta_0": torch.tensor(
        [0.3500, 0.3500, 0.3500, 0.3500, 0.3500]
    ),
    "parameter_intervention_value_var1_0": torch.tensor(
        [0.3500, 0.3500, 0.3500, 0.3500, 0.3500]
    ),
    "parameter_intervention_value_var1_1": torch.tensor(
        [0.3500, 0.3500, 0.3500, 0.3500, 0.3500]
    ),
    "parameter_intervention_value_var2_20": torch.tensor(
        [0.3500, 0.3500, 0.3500, 0.3500, 0.3500]
    ),
}


# --- Intervention weaving utilities ----
@pytest.mark.parametrize("intervention", intervention_values.keys())
def test_get_times_for(intervention):
    times = result_processing.get_times_for(intervention, intervention_times)
    time_code = intervention.split("_")[-1]
    expected_value = float(time_code)

    assert times is not None, "Time not found when expected"
    assert float(times[0]) == expected_value, "Wrong value found"

    with pytest.raises(KeyError):
        result_processing.get_times_for(intervention, {})

    with pytest.raises(KeyError):
        result_processing.get_times_for(intervention, {"not_a_real_time_10": []})

    with pytest.raises(KeyError):
        result_processing.get_times_for(
            intervention, {"parameter_intervention_time_10": []}
        )

    with pytest.raises(ValueError):
        result_processing.get_times_for(
            intervention,
            {
                f"parameter_intervention_time_{time_code}": [],
                f"parameter_sillyness_time_{time_code}": [],
            },
        )


@pytest.mark.parametrize("name", ["underscored", "with_underscore", "I", "i"])
def test_find_target_col(name):
    good_columns = [
        "before_underscored_param",
        "underscored_after_state",
        "sample_with_underscore_state",
        "i_state",
        "sampli_id_state",
        "persistent_I_param",
    ]
    result = result_processing.find_target_col(name, good_columns)
    assert name in result
    multiple_match_columns = [
        "i_state",
        "persistent_i_param",
        "before_underscored_param",
        "underscored_param",
        "with_underscore_param",
        "not_with_underscore_state",
        "With_I_param",
        "I_state",
    ]
    with pytest.raises(ValueError):
        result_processing.find_target_col(name, multiple_match_columns)
    no_match_columns = [
        "stuff_I_stuff_state",
        "sampli_state",
        "before_with_underscore_after_param",
        "underscored_after_state",
    ]
    with pytest.raises(KeyError):
        result_processing.find_target_col(name, no_match_columns)


@pytest.mark.parametrize("logging_step_size", [1, 5, 10, 12, 23])
def test_set_intervention_values(logging_step_size):
    model_1_path = (
        "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration"
        "/main/data/models/SEIRHD_NPI_Type1_petrinet.json"
    )

    num_samples = 3
    sample = pyciemss.sample(
        model_1_path,
        end_time=100,
        logging_step_size=logging_step_size,
        num_samples=num_samples,
        start_time=0.0,
        solver_method="euler",
        solver_options={"step_size": 0.1},
        time_unit="nominal",
    )

    intervention_values = {
        "parameter_intervention_value_beta_c_0": torch.tensor([0.0, 100.0, 200.0])
    }

    raw_internention_times = [logging_step_size * (n + 1) for n in range(num_samples)]

    intervention_times = {
        "parameter_intervention_time_0": torch.tensor(raw_internention_times)
    }
    intervention = "parameter_intervention_value_beta_c_0"
    df = result_processing.set_intervention_values(
        sample["data"],
        intervention,
        intervention_values[intervention],
        intervention_times,
    )

    for name, group in df.groupby("sample_id"):
        group = group.set_index("timepoint_nominal")
        time = raw_internention_times[name]
        expected = name * 100

        if time - logging_step_size > 0:
            found = group.iloc[0].at["persistent_beta_c_param"]
            assert (
                found != expected
            ), f"First step did not expect {expected} but found it (at group {name} time 0)"

            found = group.loc[time - logging_step_size].at["persistent_beta_c_param"]
            assert (
                found != expected
            ), f"Pre-intervention did not expect {expected} but found it (at group {name} time 0)"

        found = group.loc[time].at["persistent_beta_c_param"]
        assert (
            found == expected
        ), f"Post-intervention {expected} but found {found} (at group {name} time {time})"

        found = group.iloc[-1].at["persistent_beta_c_param"]
        assert (
            found == expected
        ), f"Long post-intervention {expected} but found {found} (at group {name} at last time)"
