import torch
from torch import nn, Tensor
from typing import Dict, Union, Callable

class Event(nn.Module):
    '''
    Base class for all events in the ODE solver.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, t: float, state: tuple[float, ...]) -> float: 
        raise NotImplementedError

class StaticEvent(Event):
    '''
    Use this event type to represent a static event in the ODE solver.
    Base class for ObservationEvent, StartEvent, and LoggingEvent.
    '''

    def __init__(self, time: float):
        self.time = torch.as_tensor(time)
        super().__init__()

    def __repr__(self):
        return f"{self.__class__.__name__}(time={self.time})"

    def __lt__(self, other: Event) -> bool:
        return self.time < other.time
    
    def forward(self, t: Tensor, state: tuple[Tensor, ...]) -> Tensor:
        return t - self.time

class ObservationEvent(StaticEvent):
    '''
    Use this event type to represent an observation at a given time.
    This is used in the ODE solver to trigger likelihood evaluations.
    '''
    def __init__(self, time: float, observation: Dict[str, float]):
        # self.var_name = var_name
        self.observation = {k: torch.as_tensor(v) for k, v in observation.items()}
        super().__init__(time)

    def __repr__(self):
        return f"{self.__class__.__name__}(time={self.time}, observation={self.observation})"

class StartEvent(StaticEvent):
    '''
    Use this event type to start the trajectory at a given time with a given initial state.    
    '''
    def __init__(self, time: float, initial_state: Dict[str, float]):
        self.initial_state = {k: torch.as_tensor(v) for k, v in initial_state.items()}
        super().__init__(time)

    def __lt__(self, other: Event) -> bool:
        # Start events are always first
        return self.time <= other.time

    def __repr__(self):
        return f"{self.__class__.__name__}(time={self.time}, initial_state={self.initial_state})"

class LoggingEvent(StaticEvent):
    '''
    Use this event type to measure the state of the system at a given time.
    This will not trigger an observation, nor will it stop the trajectory.
    '''
    def __init__(self, time: float):
        super().__init__(time)

class StaticParameterInterventionEvent(StaticEvent):
    '''
    Use this event type to represent a static parameter intervention at a given time.
    '''

    def __init__(self, time: float, parameter: str, value: float):
        self.parameter = parameter
        self.value = torch.as_tensor(value)
        super().__init__(time)

    def __repr__(self):
        return f"{self.__class__.__name__}(time={self.time}, parameter={self.parameter}, value={self.value})"

class DynamicEvent(Event):
    '''
    Use this event type to represent a dynamic event in the ODE solver.
    This will be the base class for state-dependent interventions.
    '''
    def __init__(self, stop_condition: nn.Module):
        self.stop_condition = stop_condition
        super().__init__()

    def forward(self, t: Tensor, state: tuple[Tensor, ...]) -> Tensor:
        raise NotImplementedError
    
