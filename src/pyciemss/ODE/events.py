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

class MeasurementEvent(Event):
    # Use this event type to measure the state of the system at a given time.
    # This will not trigger an observation, nor will it stop the trajectory.
    def __init__(self, time: Tensor):
        self.time = time

    def forward(self, t: Tensor, state: tuple[Tensor, ...]) -> Tensor:
        return t - self.time

class DynamicStopEvent(Event):
    def forward(self, t: Tensor, state: tuple[Tensor, ...]) -> Tensor:
        raise NotImplementedError