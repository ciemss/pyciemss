import os
from collections.abc import Mapping
from typing import Any, Dict, Optional, TypeVar

import numpy as np
import torch

from pyciemss.ouu.qoi import obs_nday_average_qoi

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
]

REGNET_MODELS = [
    ModelFixture(
        os.path.join(MODELS_PATH, "LV_rabbits_wolves_model02_regnet.json"), "beta"
    ),
    ModelFixture(
        os.path.join(MODELS_PATH, "LV_rabbits_wolves_model03_regnet.json"), "beta"
    ),
    ModelFixture(os.path.join(MODELS_PATH, "LV_goat_chupacabra_regnet.json"), "beta"),
]

STOCKFLOW_MODELS = [
    ModelFixture(os.path.join(MODELS_PATH, "SEIRD_stockflow.json"), "p_cbeta"),
    ModelFixture(os.path.join(MODELS_PATH, "SEIRHDS_stockflow.json"), "p_cbeta"),
    ModelFixture(os.path.join(MODELS_PATH, "SEIRHD_stockflow.json"), "p_cbeta"),
]

optimize_kwargs_SIRstockflow = {
    "qoi": lambda x: obs_nday_average_qoi(x, ["I_state"], 1),
    "risk_bound": 300.0,
    "static_parameter_interventions": {torch.tensor(1.0): "p_cbeta"},
    "objfun": lambda x: np.abs(0.35 - x),
    "initial_guess_interventions": 0.15,
    "bounds_interventions": [[0.1], [0.5]],
}

OPT_MODELS = [
    ModelFixture(
        os.path.join(MODELS_PATH, "SIR_stockflow.json"),
        important_parameter="p_cbeta",
        optimize_kwargs=optimize_kwargs_SIRstockflow,
    ),
]

MODELS = PETRI_MODELS + REGNET_MODELS + STOCKFLOW_MODELS

MODEL_URLS = [model.url for model in MODELS]

START_TIMES = [0.0]
END_TIMES = [40.0]

LOGGING_STEP_SIZES = [5.0]

NUM_SAMPLES = [2]
NON_POS_INTS = [
    3.5,
    -3,
    0,
    torch.tensor(3),
]  # bad candidates for num_samples/num_iterations


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
            torch.arange(start_time + logging_step_size, end_time, logging_step_size)
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
