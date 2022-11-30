from typing import Iterable, Union, Dict

import pyro
import torch

import pyro.distributions as dist

from pyciemss.ODE.abstractions import ObservationModel

class GaussianNoise(ObservationModel):
    def __init__(self, noise_variance: float, solution_var_names: Iterable[str]) -> None:
        super().__init__()
        self.noise_variance = noise_variance
        self.solution_var_names = solution_var_names

    def forward(self, solution: torch.Tensor, data: Union[None, Dict[str, Union[None, torch.Tensor]]]) -> torch.Tensor:

        if data == None:
            data = {n: None for n in self.solution_var_names}
        
        observations = [pyro.sample(self.solution_var_names[i], dist.Normal(solution[:, i], self.noise_variance).to_event(1), obs=data[self.solution_var_names[i]]) for i in range(solution.shape[1])]

        return observations