import pyro

from typing import TypeVar, Optional
import functools

# Declare types
# Note: this doesn't really do anything. More of a placeholder for how derived classes should be declared.
Data                   = TypeVar('Data')
Intervention           = TypeVar('Intervention')
InferredParameters     = TypeVar('InferredParameters')
State                  = TypeVar('State')
Simulation             = TypeVar('Simulation')
Variable               = TypeVar('Variable')
ObjectiveFunction      = TypeVar('ObjectiveFunction')
Constraints            = TypeVar('Constraints')
OptimizationAlgorithm  = TypeVar('OptimizationAlgorithm')
OptimizationResult     = TypeVar('OptimizationResult')

# TODO: Figure out how to declare the parameteric type of `DynamicalSystem` in the signature.
class DynamicalSystem(pyro.nn.PyroModule):
    '''
    Base class for dynamical systems.
    '''
    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        raise NotImplementedError  
     
# TODO: Figure out how to declare the parameteric type of `DynamicalSystem` in the signature.
@functools.singledispatch
def setup_model(model: DynamicalSystem, *args, **kwargs) -> DynamicalSystem:
    
    '''
    Instatiate a model for a particular configuration of initial conditions, boundary conditions, logging events, etc.
    '''
    raise NotImplementedError

# TODO: Figure out how to declare the parameteric type of `DynamicalSystem` in the signature.
@functools.singledispatch
def reset_model(model: DynamicalSystem, *args, **kwargs) -> DynamicalSystem:
    '''
    Reset a model to its initial state.
    reset_model * setup_model = id
    '''
    raise NotImplementedError

# TODO: Figure out how to declare the parameteric type of `DynamicalSystem` in the signature.
@functools.singledispatch
def intervene(model: DynamicalSystem, intervention: Intervention, *args, **kwargs) -> DynamicalSystem:
    '''
    Intervene on a model.
    '''
    raise NotImplementedError

# TODO: Figure out how to declare the parameteric type of `DynamicalSystem` in the signature.
@functools.singledispatch
def calibrate(model: DynamicalSystem, data: Data, *args, **kwargs) -> InferredParameters:
    '''
    Approximate the posterior distribution over DynamicalSystem parameters conditional on data.
    '''
    raise NotImplementedError

# TODO: Figure out how to declare the parameteric type of `DynamicalSystem` in the signature.
@functools.singledispatch
def sample(model: DynamicalSystem, inferred_parameters: Optional[InferredParameters] = None, *args, **kwargs) -> Simulation:
    '''
    Sample `num_samples` trajectories from the prior distribution over ODE models.
    '''
    raise NotImplementedError

# TODO: Figure out how to declare the parameteric type of `DynamicalSystem` in the signature.
@functools.singledispatch
def optimize(model: DynamicalSystem, 
                 objective_function: ObjectiveFunction,
                 constraints: Constraints,
                 optimization_algorithm: OptimizationAlgorithm,
                 *args, **kwargs) -> OptimizationResult:
    '''
    Optimize the objective function subject to the constraints.
    '''
    raise NotImplementedError