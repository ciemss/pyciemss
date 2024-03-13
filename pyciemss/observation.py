from typing import Callable, Dict, Optional, Set

import pyro
import torch


class NoiseModel(pyro.nn.PyroModule):
    """
    An NoiseModel is a function that takes a state and returns a state sampled from some pyro distribution.
    """

    def __init__(self, vars: Set[str] = set()):
        super().__init__()
        self.vars = vars

    def forward(self, state: Dict[str, torch.Tensor]) -> None:
        raise NotImplementedError


class StateIndependentNoiseModel(NoiseModel):
    def __init__(
        self,
        vars: Set[str] = set(),
        observables: Optional[
            Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
        ] = None,
    ):
        super().__init__(vars=vars)
        self.observables = observables

    def markov_kernel(
        self, name: str, val: torch.Tensor
    ) -> pyro.distributions.Distribution:
        raise NotImplementedError

    def forward(self, state: Dict[str, torch.Tensor]) -> None:
        if self.observables is not None:
            for k, v in self.observables(state).items():
                state[k] = v

        for k in self.vars:
            pyro.sample(
                f"{k}_noisy",
                self.markov_kernel(k, state[k]),
            )


class NormalNoiseModel(StateIndependentNoiseModel):
    def __init__(
        self,
        vars: Set[str] = set(),
        observables: Optional[
            Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
        ] = None,
        scale: float = 1.0,
    ):
        super().__init__(vars=vars, observables=observables)
        self.scale = scale

    def markov_kernel(
        self, name: str, val: torch.Tensor
    ) -> pyro.distributions.Distribution:
        return pyro.distributions.Normal(val, self.scale * torch.abs(val)).to_event(1)
