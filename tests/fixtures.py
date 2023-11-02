from typing import Dict, TypeVar

import torch

T = TypeVar("T")

# SEE https://github.com/DARPA-ASKEM/Model-Representations/issues/62 for discussion of valid models.

PETRI_URLS = [
    # "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/flux_typed.json",
    # "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/flux_typed_aug.json",
    # "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/ont_pop_vax.json",
    "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/sir.json",
    # "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/sir_flux_span.json",
    "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/sir_typed.json",
    "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/examples/sir_typed_aug.json",
]

REGNET_URLS = [
    "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/regnet/examples/lotka_volterra.json",
    # "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/regnet/examples/syntax_edge_cases.json",
]

STOCKFLOW_URLS = [
    "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/stockflow/examples/sir.json"
]

MODEL_URLS = PETRI_URLS + REGNET_URLS + STOCKFLOW_URLS

START_TIMES = [0.0, 1.0, 2.0]
END_TIMES = [3.0, 4.0, 5.0]

LOGGING_STEP_SIZES = [0.1]

NUM_SAMPLES = [2]


def check_keys_match(obj1: Dict[str, T], obj2: Dict[str, T]):
    assert set(obj1.keys()) == set(obj2.keys()), "Objects have different variables."
    return True


def check_states_match_in_all_but_values(
    traj1: Dict[str, torch.Tensor], traj2: Dict[str, torch.Tensor]
):
    assert check_keys_match(traj1, traj2)

    for k in traj1.keys():
        if k[:5] == "state":
            assert not torch.allclose(
                traj2[k], traj1[k]
            ), f"Trajectories are identical in state trajectory of variable {k}, but should differ."

    return True


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

        if k[:5] == "state":
            assert v.shape == (
                num_samples,
                len(torch.arange(start_time, end_time, logging_step_size))
                - 1,  # Does not include start_time
            )
        else:
            assert v.shape == (num_samples,)

    return True
