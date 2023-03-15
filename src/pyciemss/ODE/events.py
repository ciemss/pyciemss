import torch
from torch import Tensor, nn
from typing import Dict

class Event(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t: Tensor, state: tuple[Tensor, ...]) -> Tensor: 
        raise NotImplementedError

class StaticEvent(Event):
    def __init__(self, time: Tensor):
        self.time = time
        super().__init__()

    def __repr__(self):
        return f"{self.__class__.__name__}(time={self.time})"

    def __lt__(self, other: Event) -> bool:
        return self.time < other.time
    
    def forward(self, t: Tensor, state: tuple[Tensor, ...]) -> Tensor:
        return t - self.time

class ObservationEvent(StaticEvent):
    def __init__(self, time: Tensor, var_name: str, observation: Tensor):
        self.var_name = var_name
        self.observation = observation
        super().__init__(time)

    def __repr__(self):
        return f"{self.__class__.__name__}(time={self.time}, var_name={self.var_name}, observation={self.observation})"

class StartEvent(StaticEvent):
    def __init__(self, time: Tensor, initial_state: Dict[str, Tensor]):
        self.initial_state = initial_state
        super().__init__(time)

    def __repr__(self):
        return f"{self.__class__.__name__}(time={self.time}, initial_state={self.initial_state})"

class LoggingEvent(StaticEvent):
    # Use this event type to measure the state of the system at a given time.
    # This will not trigger an observation, nor will it stop the trajectory.
    def __init__(self, time: Tensor):
        super().__init__(time)

class DynamicStopEvent(Event):
    def __init__(self, stop_condition: nn.Module):
        self.stop_condition = stop_condition
        super().__init__()

    def forward(self, t: Tensor, state: tuple[Tensor, ...]) -> Tensor:
        raise NotImplementedError