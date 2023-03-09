import torch
from torch import Tensor, nn
from typing import Dict

class Event(nn.Module):
    def forward(self, t: Tensor, state: tuple[Tensor, ...]) -> Tensor: 
        raise NotImplementedError

class ObservationEvent(Event):
    def __init__(self, time: Tensor, observation: Dict[str, Tensor]):
        self.time = time
        self.observation = observation

    def forward(self, t: Tensor, state: tuple[Tensor, ...]) -> Tensor:
        return t - self.time

class StaticStopEvent(Event):
    def __init__(self, time: Tensor):
        self.time = time

    def forward(self, t: Tensor, state: tuple[Tensor, ...]) -> Tensor:
        return t - self.time
    
class DynamicStopEvent(Event):
    def forward(self, t: Tensor, state: tuple[Tensor, ...]) -> Tensor:
        raise NotImplementedError