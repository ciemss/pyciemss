from typing import Dict

import pyro
import torch


class NoiseModel(pyro.nn.PyroModule):
    """
    An NoiseModel is a function that takes a state and returns a state sampled from some pyro distribution.
    """

    def forward(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class NormalNoiseModel(NoiseModel):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            k: pyro.sample(
                f"{k}_observed",
                pyro.distributions.Normal(
                    state[k], torch.ones_like(state[k]) * self.scale
                ),
            )
            for k in state.keys()
        }
