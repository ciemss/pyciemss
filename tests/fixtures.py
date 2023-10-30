from typing import TypeVar

import torch
from chirho.dynamical.ops import State

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
    # "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/regnet/examples/lotka_volterra.json",
    # "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/regnet/examples/syntax_edge_cases.json",
]

STOCKFLOW_URLS = [
    # "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/stockflow/examples/sir.json"
]

MODEL_URLS = PETRI_URLS + REGNET_URLS + STOCKFLOW_URLS

START_TIMES = [torch.tensor(0.0), torch.tensor(1.0), torch.tensor(2.0)]
END_TIMES = [torch.tensor(3.0), torch.tensor(4.0), torch.tensor(5.0)]

LOGGING_STEP_SIZES = [torch.tensor(0.09)]


def check_keys_match(obj1: State[T], obj2: State[T]):
    assert set(obj1.keys()) == set(obj2.keys()), "Objects have different variables."
    return True


def check_states_match(state1: State[torch.Tensor], state2: State[torch.Tensor]):
    assert check_keys_match(state1, state2)

    for k in state1.keys():
        assert torch.allclose(
            state1[k], state2[k]
        ), f"Trajectories differ in state trajectory of variable {k}, but should be identical."

    return True


def check_states_match_in_all_but_values(
    traj1: State[torch.Tensor], traj2: State[torch.Tensor]
):
    assert check_keys_match(traj1, traj2)

    for k in traj1.keys():
        assert not torch.allclose(
            traj2[k], traj1[k]
        ), f"Trajectories are identical in state trajectory of variable {k}, but should differ."

    return True
