from typing import Dict, Set

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
    def __init__(self, vars: Set[str] = set()):
        super().__init__(vars=vars)

    def markov_kernel(
        self, name: str, val: torch.Tensor
    ) -> pyro.distributions.Distribution:
        raise NotImplementedError

    def forward(self, state: Dict[str, torch.Tensor]) -> None:
        for k in self.vars:
            pyro.sample(
                f"{k}_noisy",
                self.markov_kernel(k, state[k]),
            )


class NormalNoiseModel(StateIndependentNoiseModel):
    def __init__(self, vars: Set[str] = set(), scale: float = 1.0):
        super().__init__(vars=vars)
        self.scale = scale

    def markov_kernel(
        self, name: str, val: torch.Tensor
    ) -> pyro.distributions.Distribution:
        return pyro.distributions.Normal(val, self.scale).to_event(1)
