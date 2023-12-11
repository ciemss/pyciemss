from typing import Dict, TypeVar

import torch

T = TypeVar("T")

# See https://github.com/DARPA-ASKEM/Model-Representations/issues/62 for discussion of valid models.

SIM_INT_DIR = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main"

PETRI_URLS = [
    f"{SIM_INT_DIR}/data/models/SEIRD_base_model01_petrinet.json",
    f"{SIM_INT_DIR}/data/models/SEIRHD_NPI_Type1_petrinet.json",
    f"{SIM_INT_DIR}/data/models/SEIRHD_NPI_Type2_petrinet.json",
    f"{SIM_INT_DIR}/data/models/SEIRHD_base_model01_petrinet.json",
    f"{SIM_INT_DIR}/data/models/SEIRHD_with_reinfection01_petrinet.json",  # noqa: E501
]

REGNET_URLS = [
    "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/regnet/examples/lotka_volterra.json",
]

STOCKFLOW_URLS = [
    f"{SIM_INT_DIR}/data/models/SEIRD_stockflow.json",
    f"{SIM_INT_DIR}/data/models/SEIRHDS_stockflow.json",
    f"{SIM_INT_DIR}/data/models/SEIRHD_stockflow.json",
    f"{SIM_INT_DIR}/data/models/SEIR_stockflow.json",
    f"{SIM_INT_DIR}/data/models/SIR_stockflow.json",
]

MODEL_URLS = PETRI_URLS + REGNET_URLS + STOCKFLOW_URLS

START_TIMES = [0.0, 10.0]
END_TIMES = [30.0, 40.0]

LOGGING_STEP_SIZES = [1.0]

NUM_SAMPLES = [2]


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
    traj1: Dict[str, torch.Tensor], traj2: Dict[str, torch.Tensor]
):
    assert check_keys_match(traj1, traj2)

    for k, val in traj1.items():
        if val.ndim == 2:
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

        if v.ndim == 2:
            assert v.shape == (num_samples, num_timesteps)
        else:
            assert v.shape == (num_samples,)

    return True
