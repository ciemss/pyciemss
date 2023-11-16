from typing import Dict, TypeVar

import torch

T = TypeVar("T")

# SEE https://github.com/DARPA-ASKEM/Model-Representations/issues/62 for discussion of valid models.

PETRI_URLS = [
    "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/raw_models/SEIRD_base_model01.json",
    "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/raw_models/SEIRHD_NPI_Type1.json",
    "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/raw_models/SEIRHD_NPI_Type2.json",
    "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/raw_models/SEIRHD_base_model01.json",
    "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/raw_models/SEIRHD_three_beta.json",
    "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/raw_models/SEIRHD_two_beta.json",
    "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/raw_models/SEIRHD_with_reinfection01.json",  # noqa: E501
]

REGNET_URLS = [
    # "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/regnet/examples/lotka_volterra.json",
]

STOCKFLOW_URLS = [
    # "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/stockflow/examples/sir.json",
]

MODEL_URLS = PETRI_URLS + REGNET_URLS + STOCKFLOW_URLS

START_TIMES = [0.0, 10.0]
END_TIMES = [30.0, 40.0]

LOGGING_STEP_SIZES = [1.0]

NUM_SAMPLES = [2]


def check_keys_match(obj1: Dict[str, T], obj2: Dict[str, T]):
    assert set(obj1.keys()) == set(obj2.keys()), "Objects have different variables."
    return True


def check_states_match(
    traj1: Dict[str, torch.Tensor], traj2: Dict[str, torch.Tensor], postfix="state"
):
    assert check_keys_match(traj1, traj2)

    for k in traj1.keys():
        if k[-len(postfix) :] == postfix:
            assert torch.allclose(
                traj2[k], traj1[k]
            ), f"Trajectories differ in state trajectory of variable {k}."

    return True


def check_states_match_in_all_but_values(
    traj1: Dict[str, torch.Tensor], traj2: Dict[str, torch.Tensor], postfix="state"
):
    assert check_keys_match(traj1, traj2)

    for k in traj1.keys():
        if k[-len(postfix) :] == postfix:
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

        if k[-5:] == "state" or k[-8:] == "observed":
            assert v.shape == (
                num_samples,
                len(
                    torch.arange(
                        start_time + logging_step_size, end_time, logging_step_size
                    )
                ),
            )
        else:
            assert v.shape == (num_samples,)

    return True
