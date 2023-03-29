import torch
import pyro

from pyro.infer import Predictive
# from pyro.infer.autoguide.guides import AutoNormal
# from pyro.poutine import block

# from pyciemss.risk.ouu import solveOUU

# from pyciemss.utils.inference_utils import run_inference

from typing import TypeVar, Iterable, Optional, Union
import functools

# Declare types
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


@functools.singledispatch
def compile_pp(model_or_path, *args, **kwargs) -> DynamicalSystem:
    '''
    Loads a model from a file or a string and compiles it into a dynamical system
    '''
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
def condition(model: DynamicalSystem, data: Data, *args, **kwargs) -> DynamicalSystem:
    '''
    Condition a model on observed data.
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
def sample(model: DynamicalSystem, inferred_parameters: Optional[InferredParameters] = None, *args, **kwargs) -> Simulation:
    '''
    Sample `num_samples` trajectories from the prior distribution over ODE models.
    '''
    raise NotImplementedError

# TODO: Figure out how to declare the parameteric type of `DynamicalSystem` in the signature.
@functools.singledispatch
def calibrate(model: DynamicalSystem, *args, **kwargs) -> InferredParameters:
    '''
    Approximate the posterior distribution over DynamicalSystem parameters.
    If the model has not been conditioned on data, this will be the prior distribution.
    '''
    raise NotImplementedError

# TODO: Figure out how to declare the parameteric type of `DynamicalSystem` in the signature.
@functools.singledispatch
def optimization(model: DynamicalSystem, 
                 objective_function: ObjectiveFunction,
                 constraints: Constraints,
                 optimization_algorithm: OptimizationAlgorithm,
                 *args, **kwargs) -> OptimizationResult:
    '''
    Optimize the objective function subject to the constraints.
    '''
    raise NotImplementedError