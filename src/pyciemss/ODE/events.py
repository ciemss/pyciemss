import torch
from torch import Tensor, nn

class Event(nn.Module):
    def __init__():
        raise NotImplementedError
    
    def forward(self, t: Tensor, state: tuple[Tensor, ...]) -> Tensor: 
        raise NotImplementedError

class ObservationEvent(Event):
    def __init__(self, time: Tensor, observation: Tensor):
        self.time = time
        self.observation = observation

    def forward(self, t: Tensor, state: tuple[Tensor, ...]) -> Tensor:
        return t - self.time
