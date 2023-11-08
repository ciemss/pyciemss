from __future__ import annotations

import functools
from typing import Callable, Dict, List, TypeVar

import pyro
import torch
from chirho.dynamical.handlers.solver import Solver, TorchDiffEq
from chirho.dynamical.ops import State
from pyro.contrib.autoname import scope

from pyciemss.compiled_dynamics import CompiledDynamics

T = TypeVar("T")


class EnsembleCompiledDynamics(pyro.nn.PyroModule):
    def __init__(
        self,
        dynamics_models: List[CompiledDynamics],
        dirichlet_alpha: torch.Tensor,
        solution_mappings: List[
            Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
        ],
    ):
        super().__init__()
        self.dynamics_models = dynamics_models
        self.register_buffer("dirichlet_alpha", dirichlet_alpha)
        self.solution_mappings = solution_mappings

    @pyro.nn.PyroSample
    def model_weights(self):
        return pyro.distributions.Dirichlet(self.dirichlet_alpha)

    def forward(
        self,
        start_time: torch.Tensor,
        end_time: torch.Tensor,
        solver: Solver = TorchDiffEq(),
    ) -> State[torch.Tensor]:
        solutions = [State()] * len(self.dynamics_models)
        for i, dynamics in enumerate(self.dynamics_models):
            with scope(prefix=f"model_{i}"):
                solutions[i] = dynamics(start_time, end_time, solver)

        return State(
            **{
                k: sum([self.model_weights[i] * v[k] for i, v in enumerate(solutions)])
                for k in solutions[0].keys()
            }
        )

    @functools.singledispatchmethod
    @classmethod
    def load(cls, obj, *args, **kwargs) -> "EnsembleCompiledDynamics":
        raise NotImplementedError(f"Cannot load object of type {type(obj)}.")

    @load.register
    @classmethod
    def _load_from_list(cls, srcs: list, dirichlet_alpha, solution_mappings):
        dynamics_models = [CompiledDynamics.load(src) for src in srcs]
        return cls(dynamics_models, dirichlet_alpha, solution_mappings)
