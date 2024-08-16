import os
from collections.abc import Mapping
from typing import Any, Dict, Optional, TypeVar

import numpy as np
import pandas as pd
import torch

from pyciemss.integration_utils.intervention_builder import (
    intervention_func_combinator,
    param_value_objective,
    start_time_objective,
    start_time_param_value_objective,
)
from pyciemss.ouu.qoi import obs_max_qoi, obs_nday_average_qoi

T = TypeVar("T")

MODELS_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/models/"
PDE_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/pde-petri-amrs/petrinet/"
DATA_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/datasets/"


class ModelFixture:
    def __init__(
        self,
        url: str,
        important_parameter: Optional[str] = None,
        data_path: Optional[str] = None,
        data_mapping: Dict[str, str] = {},
        data_mapped_to_observable: bool = False,
        optimize_kwargs: Dict[str, Any] = None,
    ):
        self.url = url
        self.important_parameter = important_parameter
        self.data_path = data_path
        self.data_mapping = data_mapping
        self.data_mapped_to_observable = data_mapped_to_observable
        self.optimize_kwargs = optimize_kwargs


# See https://github.com/DARPA-ASKEM/Model-Representations/issues/62 for discussion of valid models.

PETRI_MODELS = [
    ModelFixture(
        os.path.join(MODELS_PATH, "SEIRHD_NPI_Type1_petrinet.json"),
        "gamma",
        os.path.join(DATA_PATH, "traditional.csv"),
        {"Infected": "I"},
        False,
    ),
    ModelFixture(
        os.path.join(MODELS_PATH, "SEIRHD_NPI_Type2_petrinet.json"),
        "kappa",
        os.path.join(DATA_PATH, "SIR_data_case_hosp.csv"),
        {"case": "infected", "hosp": "hospitalized"},
        True,
    ),
    ModelFixture(
        os.path.join(MODELS_PATH, "SEIRHD_with_reinfection01_petrinet.json"),
        "beta",
        os.path.join(DATA_PATH, "SIR_data_case_hosp.csv"),
        {"case": "infected", "hosp": "hospitalized"},
        True,
    ),
    ModelFixture(
        os.path.join(
            PDE_PATH, "examples/pde/advection/advection_backward_1_0.01_3.json"
        ),
        "u",
    ),
    ModelFixture(
        os.path.join(MODELS_PATH, "SIR_param_in_observables.json"),
        "beta",
        os.path.join(DATA_PATH, "SIR_data_case_hosp.csv"),
        {"case": "incident_cases", "hosp": "I"},
        True,
    ),
]

REGNET_MODELS = [
    ModelFixture(
        os.path.join(MODELS_PATH, "LV_rabbits_wolves_model02_regnet.json"), "beta"
    ),
    ModelFixture(
        os.path.join(MODELS_PATH, "LV_rabbits_wolves_model03_regnet.json"), "beta"
    ),
    # ModelFixture(os.path.join(MODELS_PATH, "LV_goat_chupacabra_regnet.json"), "beta"),
]

STOCKFLOW_MODELS = [
    ModelFixture(os.path.join(MODELS_PATH, "SEIRD_stockflow.json"), "p_cbeta"),
    ModelFixture(os.path.join(MODELS_PATH, "SEIRHDS_stockflow.json"), "p_cbeta"),
    ModelFixture(os.path.join(MODELS_PATH, "SEIRHD_stockflow.json"), "p_cbeta"),
]

optkwargs_SIRstockflow_param = {
    "qoi": lambda x: obs_nday_average_qoi(x, ["I_state"], 1),
    "risk_bound": 300.0,
    "static_parameter_interventions": param_value_objective(
        param_name=["p_cbeta"],
        param_value=[lambda x: torch.tensor([x])],
        start_time=[torch.tensor(1.0)],
    ),
    "objfun": lambda x: np.abs(0.35 - x),
    "initial_guess_interventions": 0.15,
    "bounds_interventions": [[0.1], [0.5]],
}

optkwargs_SIRstockflow_time = {
    "qoi": lambda x: obs_nday_average_qoi(x, ["I_state"], 1),
    "risk_bound": 300.0,
    "static_parameter_interventions": start_time_objective(
        param_name=["p_cbeta"],
        param_value=[torch.tensor([0.15])],
    ),
    "objfun": lambda x: -x,
    "initial_guess_interventions": 1.0,
    "bounds_interventions": [[0.0], [40.0]],
}

optkwargs_SIRstockflow_time_param = {
    "qoi": lambda x: obs_nday_average_qoi(x, ["I_state"], 1),
    "risk_bound": 300.0,
    "static_parameter_interventions": start_time_param_value_objective(
        param_name=["p_cbeta"],
    ),
    "objfun": lambda x: -x[0] * 0.25 / (0.0 - 40.0) + np.abs(0.35 - x[1]) * 1.0,
    "initial_guess_interventions": [1.0, 0.15],
    "bounds_interventions": [[0.0, 0.1], [40.0, 0.5]],
}

# Creating a combined interventions by combining into list of Callables
intervened_params = ["beta_c", "gamma"]
static_parameter_interventions1 = param_value_objective(
    param_name=[intervened_params[0]],
    start_time=torch.tensor([10.0]),
)
static_parameter_interventions2 = start_time_objective(
    param_name=[intervened_params[1]],
    param_value=torch.tensor([0.45]),
)
optkwargs_SEIRHD_paramtimeComb_maxQoI = {
    "qoi": lambda x: obs_max_qoi(x, ["I_state"]),
    "risk_bound": 3e5,
    "static_parameter_interventions": intervention_func_combinator(
        [static_parameter_interventions1, static_parameter_interventions2],
        [1, 1],
    ),
    "objfun": lambda x: np.abs(0.35 - x[0]) - x[1],
    "initial_guess_interventions": [0.35, 5.0],
    "bounds_interventions": [[0.1, 1.0], [0.5, 90.0]],
    "fixed_static_parameter_interventions": {10.0: {"hosp": torch.tensor(0.1)}},
}

OPT_MODELS = [
    ModelFixture(
        os.path.join(MODELS_PATH, "SIR_stockflow.json"),
        optimize_kwargs=optkwargs_SIRstockflow_param,
    ),
    ModelFixture(
        os.path.join(MODELS_PATH, "SIR_stockflow.json"),
        optimize_kwargs=optkwargs_SIRstockflow_time,
    ),
    ModelFixture(
        os.path.join(MODELS_PATH, "SIR_stockflow.json"),
        optimize_kwargs=optkwargs_SIRstockflow_time_param,
    ),
    ModelFixture(
        os.path.join(MODELS_PATH, "SEIRHD_NPI_Type1_petrinet.json"),
        optimize_kwargs=optkwargs_SEIRHD_paramtimeComb_maxQoI,
    ),
]

BAD_AMRS = [
    ModelFixture(
        os.path.join(MODELS_PATH, "sir-undef-var-initial.json"),
        "p_cbeta",
    ),
    ModelFixture(
        os.path.join(MODELS_PATH, "sir-undef-var-rate-law.json"),
        "p_cbeta",
    ),
]

MODELS = PETRI_MODELS + REGNET_MODELS + STOCKFLOW_MODELS

MODEL_URLS = [model.url for model in MODELS]

START_TIMES = [0.0]
END_TIMES = [40.0]

LOGGING_STEP_SIZES = [5.0]

NUM_SAMPLES = [2]
SEIRHD_NPI_STATIC_PARAM_INTERV = [{torch.tensor(10.0): {"delta": torch.tensor(0.2)}}]
NON_POS_INTS = [
    3.5,
    -3,
    0,
    torch.tensor(3),
]  # bad candidates for num_samples/num_iterations

bad_data1 = {
    "Timestamp": {0: 1.1, 1: 2.2, 2: 3.3},
    "case": {0: 15.0, 1: "", 2: 20.0},
    "hosp": {0: 0.1, 1: 1.0, 2: 2.2},
}
bad_data2 = {
    "Timestamp": {0: 1.1, 1: 2.2, 2: 3.3},
    "case": {0: 15.0, 1: "apple", 2: 20.0},
    "hosp": {0: 0.1, 1: 1.0, 2: 2.2},
}
bad_data3 = {
    "Timestamp": {0: 1.1, 1: 2.2, 2: 3.3},
    "case": {0: 15.0, 1: " ", 2: 20.0},
    "hosp": {0: 0.1, 1: 1.0, 2: 2.2},
}
bad_data4 = {
    "Timestamp": {0: 1.1, 1: 2.2, 2: 3.3},
    "case": {0: 15.0, 1: None, 2: 20.0},
    "hosp": {0: 0.1, 1: 1.0, 2: 2.2},
}
bad_data5 = {
    "Timepoints": {0: 1.1, 1: 2.2, 2: 3.3},
    "case": {0: 15.0, 1: 18.0, 2: 20.0},
    "hosp": {0: 0.1, 1: 1.0, 2: 2.2},
}
BADLY_FORMATTED_DATAFRAMES = [
    pd.DataFrame(bad_data1),
    pd.DataFrame(bad_data2),
    pd.DataFrame(bad_data3),
    pd.DataFrame(bad_data4),
    pd.DataFrame(bad_data5),
]  # improperly formatted datasets
MAPPING_FOR_DATA_TESTS = {"case": "I", "hosp": "H"}


def check_keys_match(obj1: Dict[str, T], obj2: Dict[str, T]):
    assert set(obj1.keys()) == set(obj2.keys()), "Objects have different variables."
    return True


def check_states_match(traj1: Dict[str, torch.Tensor], traj2: Dict[str, torch.Tensor]):
    assert check_keys_match(traj1, traj2)

    for k, val in traj1.items():
        if val.ndim == 2:
            assert torch.allclose(
                traj2[k], traj1[k]
            ), f"Trajectories differ in state trajectory of variable {k}."

    return True


def check_states_match_in_all_but_values(
    traj1: Dict[str, torch.Tensor], traj2: Dict[str, torch.Tensor], state_ndim: int = 2
):
    assert check_keys_match(traj1, traj2)

    for k, val in traj1.items():
        if val.ndim == state_ndim:
            if not torch.allclose(traj2[k], traj1[k]):
                # early return, as we've already confirmed they're not identical
                return True

    assert False, "Trajectories have identical values."


def check_result_sizes(
    traj: Dict[str, torch.Tensor],
    start_time: float,
    end_time: float,
    logging_step_size: float,
    num_samples: int,
):
    for k, v in traj.items():
        assert isinstance(k, str)
        assert isinstance(v, torch.Tensor)

        num_timesteps = len(
            torch.arange(start_time, end_time + logging_step_size, logging_step_size)
        )
        if v.ndim == 2 and k == "model_weights":
            assert v.shape[0] == num_samples
        elif v.ndim == 2:
            assert v.shape == (num_samples, num_timesteps)
        else:
            assert v.shape == (num_samples,)

    return True


def check_is_state(state: torch.Tensor, value_type):
    assert isinstance(state, Mapping)

    assert all(isinstance(key, str) for key in state.keys())

    assert all(isinstance(value, value_type) for value in state.values())

    return True
