import torch

from pyro.infer import Predictive
from pyro.infer.autoguide.guides import AutoNormal
from pyro.poutine import block

from causal_pyro.query.do_messenger import do

from pyciemss.ODE.base import ODE
from pyciemss.risk.ouu import solveOUU

from pyciemss.utils.inference_utils import run_inference

from typing import TypeVar, Iterable, Optional, Union

# Declare types
PetriNet = TypeVar("PetriNet")
PriorJSON = TypeVar("PriorJSON")
ProbProg = TypeVar("ProbProg")
InferredParameters = TypeVar("InferredParameters")
State = TypeVar("State")
TSpan = TypeVar("TSpan")
Data = TypeVar("Data")
InterventionSpec = TypeVar("InterventionSpec")
Variable = TypeVar("Variable")
ObjectiveFunction = TypeVar("ObjectiveFunction")
Constraints = TypeVar("Constraints")
OptimizationAlgorithm = TypeVar("OptimizationAlgorithm")
DataCube = TypeVar("DataCube")
OptimizationResult = TypeVar("OptimizationResult")


def compile_pp(petri_G: PetriNet, prior_json: PriorJSON) -> ProbProg:

    raise NotImplementedError


def sample(
    ode_model: ProbProg,
    num_samples: int,
    initial_state: State,
    tspan: TSpan,
    inferred_parameters: Optional[InferredParameters] = None,
) -> DataCube:

    """
    Sample `num_samples` trajectories from the prior distribution over ODE models.
    """

    return Predictive(ode_model, guide=inferred_parameters, num_samples=num_samples)(
        initial_state, tspan
    )


def infer_parameters(
    ode_model: ProbProg,
    num_iterations: int,
    hidden_observations: Iterable[str],
    data: Data,
    initial_state: State,
    observed_tspan: TSpan,
) -> InferredParameters:

    """
    Use variational inference to condition `ode_model` on observed `data`.
    """

    guide = AutoNormal(block(ode_model, hide=hidden_observations))
    run_inference(
        ode_model,
        guide,
        initial_state,
        observed_tspan,
        data,
        num_iterations=num_iterations,
    )
    return guide


def intervene(ode_model: ProbProg, intervention_spec: InterventionSpec) -> ProbProg:
    return do(ode_model, intervention_spec)


def optimization(
    initial_guess: torch.tensor,
    objective_function: ObjectiveFunction,
    constraints: Constraints,
    optimizer: OptimizationAlgorithm,
):
    return solveOUU(
        initial_guess, objective_function, constraints, optimizer_algorithm=optimizer
    ).solve()
