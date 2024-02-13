from __future__ import annotations

import functools
from typing import Callable, List, Optional, TypeVar

import pyro
import torch
from chirho.dynamical.ops import State
from pyro.contrib.autoname import scope

from pyciemss.compiled_dynamics import CompiledDynamics

T = TypeVar("T")


class EnsembleCompiledDynamics(pyro.nn.PyroModule):
    def __init__(
        self,
        dynamics_models: List[CompiledDynamics],
        dirichlet_alpha: torch.Tensor,
        solution_mappings: List[Callable[[State[torch.Tensor]], State[torch.Tensor]]],
    ):
        super().__init__()
        self.dynamics_models = dynamics_models
        self.register_buffer("dirichlet_alpha", dirichlet_alpha)
        self.solution_mappings = solution_mappings

    def forward(
        self,
        start_time: torch.Tensor,
        end_time: torch.Tensor,
        logging_times: Optional[torch.Tensor] = None,
        is_traced: bool = False,
    ) -> State[torch.Tensor]:
        model_weights = pyro.sample(
            "model_weights", pyro.distributions.Dirichlet(self.dirichlet_alpha)
        )

        mapped_states: List[State[torch.Tensor]] = [dict()] * len(self.dynamics_models)

        for i, dynamics in enumerate(self.dynamics_models):
            with scope(prefix=f"model_{i}"):
                state_and_observables = dynamics(
                    start_time, end_time, logging_times, is_traced
                )
                mapped_states[i] = self.solution_mappings[i](state_and_observables)

        if not all(mapped_states[0].keys() == s.keys() for s in mapped_states):
            raise ValueError(
                "All solution mappings must return the same keys for each model."
            )

        mapped_state = dict(
            **{
                k: sum(
                    [
                        model_weights[..., i].unsqueeze(dim=-1) * v[k]
                        for i, v in enumerate(mapped_states)
                    ]
                )
                for k in mapped_states[0].keys()
            }
        )

        if is_traced:
            # Add the mapped result variables to the trace so that they can be accessed later.
            [
                pyro.deterministic(f"{name}_state", value)
                for name, value in mapped_state.items()
            ]

        return mapped_states

    @functools.singledispatchmethod
    @classmethod
    def load(cls, obj, *args, **kwargs) -> "EnsembleCompiledDynamics":
        raise NotImplementedError(f"Cannot load object of type {type(obj)}.")

    @load.register(list)
    @classmethod
    def _load_from_list(cls, srcs: list, dirichlet_alpha, solution_mappings):
        dynamics_models = [CompiledDynamics.load(src) for src in srcs]
        return cls(dynamics_models, dirichlet_alpha, solution_mappings)
