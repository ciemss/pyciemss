import pyro
import torch
from pyro.infer import Predictive

from pyciemss.PetriNetODE.base import (
    PetriNetODESystem,
    ScaledBetaNoisePetriNetODESystem,
    MiraPetriNetODESystem,
    get_name,
)
from pyciemss.risk.ouu import computeRisk, solveOUU
from pyciemss.risk.risk_measures import alpha_quantile, alpha_superquantile
from pyciemss.utils.interface_utils import convert_to_output_format, csv_to_list

import time
import numpy as np
from math import ceil

import pandas as pd

from typing import Iterable, Optional, Tuple, Union
import copy

import mira

# Load base interfaces
from pyciemss.interfaces import (
    setup_model,
    reset_model,
    intervene,
    sample,
    calibrate,
    optimize,
)

from pyciemss.PetriNetODE.events import (
    StartEvent,
    ObservationEvent,
    LoggingEvent,
    StaticParameterInterventionEvent,
)

# TODO: These interfaces should probably be just in terms of JSON-like objects.

PetriSolution = dict[str, torch.Tensor]
PetriInferredParameters = pyro.nn.PyroModule


def load_and_sample_petri_model(
    petri_model_or_path: Union[str, mira.metamodel.TemplateModel, mira.modeling.Model],
    num_samples: int,
    timepoints: Iterable[float],
    *,
    interventions: Optional[Iterable[Tuple[float, str, float]]] = None,
    start_state: Optional[dict[str, float]] = None,
    pseudocount: float = 1.0,
    start_time: float = -1e-10,
    method="dopri5",
) -> pd.DataFrame:
    """
    Load a petri net from a file, compile it into a probabilistic program, and sample from it.

    Args:
        petri_model_or_path: Union[str, mira.metamodel.TemplateModel, mira.modeling.Model]
            - A path to a petri net file, or a petri net object.
            - This path can be a URL or a local path to a mira model or AMR model.
            - Alternatively, this can be a mira template model directly.
        num_samples: int
            - The number of samples to draw from the model.
        timepoints: [Iterable[float]]
            - The timepoints to simulate the model from. Backcasting and/or forecasting is reflected in the choice of timepoints.
        interventions: Optional[Iterable[Tuple[float, str, float]]]
            - A list of interventions to apply to the model. Each intervention is a tuple of the form (time, parameter_name, value).
        start_state: Optional[dict[str, float]]
            - The initial state of the model. If None, the initial state is taken from the mira model.
        pseudocount: float > 0.0
            - The pseudocount to use for adding uncertainty to the model parameters.
            - Larger values of pseudocount correspond to more certainty about the model parameters.
        start_time: float
            - The start time of the model. This is used to align the `start_state` with the `timepoints`.
            - By default we set the `start_time` to be a small negative number to avoid numerical issues w/ collision with the `timepoints` which typically start at 0.
        method: str
            - The method to use for solving the ODE. See torchdiffeq's `odeint` method for more details.
            - If performance is incredibly slow, we suggest using `euler` to debug. If using `euler` results in faster simulation, the issue is likely that the model is stiff.

    Returns:
        samples: PetriSolution
            - The samples from the model as a pandas DataFrame.
    """
    model = load_petri_model(
        petri_model_or_path=petri_model_or_path,
        add_uncertainty=True,
        pseudocount=pseudocount,
    )

    # If the user doesn't override the start state, use the initial values from the model.
    if start_state is None:
        start_state = {
            get_name(v): v.data["initial_value"] for v in model.G.variables.values()
        }

    model = setup_model(model, start_time=start_time, start_state=start_state)

    if interventions is not None:
        intervention_events = [
            StaticParameterInterventionEvent(timepoint, parameter, value)
            for timepoint, parameter, value in interventions
        ]
        model.load_events(intervention_events)

    samples = sample(
        model,
        timepoints,
        num_samples,
        method=method,
    )
    # Fix the parameters here with reparameterize.

    processed_samples = convert_to_output_format(samples, timepoints, interventions=interventions)

    return processed_samples


def load_and_calibrate_and_sample_petri_model(
    petri_model_or_path: Union[str, mira.metamodel.TemplateModel, mira.modeling.Model],
    data_path: str,
    num_samples: int,
    timepoints: Iterable[float],
    *,
    interventions: Optional[Iterable[Tuple[float, str, float]]] = None,
    start_state: Optional[dict[str, float]] = None,
    pseudocount: float = 1.0,
    start_time: float = -1e-10,
    num_iterations: int = 1000,
    lr: float = 0.03,
    verbose: bool = False,
    num_particles: int = 1,
    autoguide=pyro.infer.autoguide.AutoLowRankMultivariateNormal,
    method="dopri5",
) -> pd.DataFrame:
    """
    Load a petri net from a file, compile it into a probabilistic program, calibrate it on data,
    and sample from the calibrated model.

    Args:
        petri_model_or_path: Union[str, mira.metamodel.TemplateModel, mira.modeling.Model]
            - A path to a petri net file, or a petri net object.
            - This path can be a URL or a local path to a mira model or AMR model.
            - Alternatively, this can be a mira template model directly.
        data_path: str
            - The path to the data to calibrate the model to. See notebook/integration_demo/data.csv for an example of the format.
        num_samples: int
            - The number of samples to draw from the model.
        timepoints: [Iterable[float]]
            - The timepoints to simulate the model from. Backcasting and/or forecasting is reflected in the choice of timepoints.
        interventions: Optional[Iterable[Tuple[float, str, float]]]
            - A list of interventions to apply to the model. Each intervention is a tuple of the form (time, parameter_name, value).
        start_state: Optional[dict[str, float]]
            - The initial state of the model. If None, the initial state is taken from the mira model.
        pseudocount: float > 0.0
            - The pseudocount to use for adding uncertainty to the model parameters.
            - Larger values of pseudocount correspond to more certainty about the model parameters.
        start_time: float
            - The start time of the model. This is used to align the `start_state` with the `timepoints`.
            - By default we set the `start_time` to be a small negative number to avoid numerical issues w/ collision with the `timepoints` which typically start at 0.
        num_iterations: int > 0
            - The number of iterations to run the calibration for.
        lr: float > 0.0
            - The learning rate to use for the calibration.
        verbose: bool
            - Whether to print out the calibration progress. This will include summaries of the evidence lower bound (ELBO) and the parameters.
        num_particles: int > 0
            - The number of particles to use for the calibration. Increasing this value will result in lower variance gradient estimates, but will also increase the computational cost per gradient step.
        autoguide: pyro.infer.autoguide.AutoGuide
            - The guide to use for the calibration. By default we use the AutoLowRankMultivariateNormal guide. This is an advanced option. Please see the Pyro documentation for more details.
        method: str
            - The method to use for the ODE solver. See `torchdiffeq.odeint` for more details.
            - If performance is incredibly slow, we suggest using `euler` to debug. If using `euler` results in faster simulation, the issue is likely that the model is stiff.

    Returns:
        samples: pd.DataFrame
            - A dataframe containing the samples from the calibrated model.
    """
    data = csv_to_list(data_path)

    model = load_petri_model(
        petri_model_or_path=petri_model_or_path,
        add_uncertainty=True,
        pseudocount=pseudocount,
    )

    # If the user doesn't override the start state, use the initial values from the model.
    if start_state is None:
        start_state = {
            get_name(v): v.data["initial_value"] for v in model.G.variables.values()
        }

    model = setup_model(model, start_time=start_time, start_state=start_state)

    if interventions is not None:
        intervention_events = [
            StaticParameterInterventionEvent(timepoint, parameter, value)
            for timepoint, parameter, value in interventions
        ]
        model.load_events(intervention_events)

    inferred_parameters = calibrate(
        model,
        data,
        num_iterations,
        lr,
        verbose,
        num_particles,
        autoguide,
        method=method,
    )
    samples = sample(
        model,
        timepoints,
        num_samples,
        inferred_parameters=inferred_parameters,
        method=method,
    )

    processed_samples = convert_to_output_format(samples, timepoints, interventions=interventions)

    return processed_samples


##############################################################################
# Internal Interfaces Below - TA4 above

def load_petri_model(
    petri_model_or_path: Union[str, mira.metamodel.TemplateModel, mira.modeling.Model],
    add_uncertainty=True,
    pseudocount=1.0,
) -> PetriNetODESystem:
    """
    Load a petri net from a file and compile it into a probabilistic program.
    """

    if add_uncertainty:
        model = ScaledBetaNoisePetriNetODESystem.from_askenet(petri_model_or_path)
        model.pseudocount = torch.tensor(pseudocount)
        return model
    else:
        return MiraPetriNetODESystem.from_askenet(petri_model_or_path)


@setup_model.register
def setup_petri_model(
    petri: PetriNetODESystem,
    start_time: float,
    start_state: dict[str, float],
) -> PetriNetODESystem:
    """
    Instantiate a model for a particular configuration of initial conditions
    """
    # TODO: Figure out how to do this without copying the petri net.
    start_event = StartEvent(start_time, start_state)
    new_petri = copy.deepcopy(petri)
    new_petri.load_event(start_event)
    return new_petri


@reset_model.register
def reset_petri_model(petri: PetriNetODESystem) -> PetriNetODESystem:
    """
    Reset a model to its initial state.
    reset_model * setup_model = id
    """
    new_petri = copy.deepcopy(petri)
    new_petri.reset()
    return new_petri


@intervene.register
def intervene_petri_model(
    petri: PetriNetODESystem, interventions: Iterable[Tuple[float, str, float]]
) -> PetriNetODESystem:
    """
    Intervene on a model.
    """
    # Note: this will have to change if we want to add more sophisticated interventions.
    interventions = [
        StaticParameterInterventionEvent(timepoint, parameter, value)
        for timepoint, parameter, value in interventions
    ]
    new_petri = copy.deepcopy(petri)
    new_petri.load_events(interventions)
    return new_petri


@calibrate.register
def calibrate_petri(
    petri: PetriNetODESystem,
    data: Iterable[Tuple[float, dict[str, float]]],
    num_iterations: int = 1000,
    lr: float = 0.03,
    verbose: bool = False,
    num_particles: int = 1,
    autoguide=pyro.infer.autoguide.AutoLowRankMultivariateNormal,
    method="dopri5",
) -> PetriInferredParameters:
    """
    Use variational inference with a mean-field variational family to infer the parameters of the model.
    """
    new_petri = copy.deepcopy(petri)
    observations = [
        ObservationEvent(timepoint, observation) for timepoint, observation in data
    ]

    for obs in observations:
        s = 0.0
        for v in obs.observation.values():
            s += v
            assert 0 <= v <= petri.total_population
        assert s <= petri.total_population or torch.isclose(s, petri.total_population)
    new_petri.load_events(observations)

    guide = autoguide(new_petri)
    optim = pyro.optim.Adam({"lr": lr})
    loss = pyro.infer.Trace_ELBO(num_particles=num_particles)
    svi = pyro.infer.SVI(new_petri, guide, optim, loss=loss)

    pyro.clear_param_store()

    for i in range(num_iterations):
        loss = svi.step(method=method)
        if verbose:
            if i % 25 == 0:
                print(f"iteration {i}: loss = {loss}")

    return guide


@sample.register
def sample_petri(
    petri: PetriNetODESystem,
    timepoints: Iterable[float],
    num_samples: int,
    inferred_parameters: Optional[PetriInferredParameters] = None,
    method="dopri5",
) -> PetriSolution:
    """
    Sample `num_samples` trajectories from the prior or posterior distribution over ODE models.
    """
    assert hasattr(
        petri, "_static_events"
    ), "Please initialize the model before sampling."
    assert len(petri._static_events) > 0, "No events initialized."
    assert isinstance(
        petri._static_events[0], StartEvent
    ), "First event should be a StartEvent."
    logging_events = [LoggingEvent(timepoint) for timepoint in timepoints]
    new_petri = copy.deepcopy(petri)
    new_petri.load_events(logging_events)
    return Predictive(new_petri, guide=inferred_parameters, num_samples=num_samples)(
        method=method
    )


@optimize.register
def optimize_petri(
    petri: PetriNetODESystem,
    timepoints: Iterable,
    interventions: dict,
    qoi: callable,
    risk_bound: float,
    objfun: callable = lambda x: np.abs(x),
    initial_guess: Iterable[float] = 0.5,
    bounds: Iterable[float] = [[0.0], [1.0]],
    inferred_parameters: Optional[PetriInferredParameters] = None,
    n_samples_ouu: int = int(1e2),
    maxiter: int = 2,
    maxfeval: int = 25,
    method="dopri5",
    roundup_decimal: int = 4,
) -> dict:
    """
    Optimization under uncertainty with risk-based constraints over ODE models.
    """
    # maxfeval: Maximum number of function evaluations for each local optimization step
    # maxiter: Maximum number of basinhopping iterations: >0 leads to multi-start
    timepoints = [float(x) for x in list(timepoints)]
    bounds = np.atleast_2d(bounds)
    u_min = bounds[0, :]
    u_max = bounds[1, :]
    # Set up risk estimation
    control_model = copy.deepcopy(petri)
    RISK = computeRisk(
        model=control_model,
        interventions=interventions,
        qoi=qoi,
        tspan=timepoints,
        risk_measure=lambda z: alpha_superquantile(z, alpha=0.95),
        num_samples=1,
        guide=inferred_parameters,
        method=method,
    )

    # Run one sample to estimate model evaluation time
    start_time = time.time()
    init_prediction = RISK.propagate_uncertainty(initial_guess)
    RISK.qoi(init_prediction)
    end_time = time.time()
    forward_time = end_time - start_time
    time_per_eval = forward_time / 1.0
    print(f"Time taken: ({forward_time/1.:.2e} seconds per model evaluation).")

    # Assign the required number of MC samples for each OUU iteration
    control_model = copy.deepcopy(petri)
    RISK = computeRisk(
        model=control_model,
        interventions=interventions,
        qoi=qoi,
        tspan=timepoints,
        risk_measure=lambda z: alpha_superquantile(z, alpha=0.95),
        num_samples=n_samples_ouu,
        guide=inferred_parameters,
        method=method,
    )
    # Define problem constraints
    constraints = (
        # risk constraint
        {"type": "ineq", "fun": lambda x: risk_bound - RISK(x)},
        # bounds on control
        {"type": "ineq", "fun": lambda x: x - u_min},
        {"type": "ineq", "fun": lambda x: u_max - x},
    )
    print(
        "Performing risk-based optimization under uncertainty (using alpha-superquantile)"
    )
    print(
        f"Estimated wait time {time_per_eval*n_samples_ouu*(maxiter+1)*maxfeval:.1f} seconds..."
    )
    start_time = time.time()
    opt_results = solveOUU(
        x0=initial_guess,
        objfun=objfun,
        constraints=constraints,
        maxiter=maxiter,
        maxfeval=maxfeval,
    ).solve()

    # Rounding up to given number of decimal places
    # TODO: move to utilities
    def round_up(num, dec=roundup_decimal):
        return ceil(num * 10**dec) / (10**dec)

    opt_results.x = round_up(opt_results.x)
    print(f"Optimization completed in time {time.time()-start_time:.2f} seconds.")
    print(f"Optimal policy:\t{opt_results.x}")

    # Check for some interventions that lead to no feasible solutions
    if opt_results.x < 0:
        print("No solution found")

    # Post-process OUU results
    print("Post-processing optimal policy...")
    tspan_plot = [float(x) for x in list(range(1, int(timepoints[-1])))]
    control_model = copy.deepcopy(petri)
    RISK = computeRisk(
        model=control_model,
        interventions=interventions,
        qoi=qoi,
        tspan=tspan_plot,
        risk_measure=lambda z: alpha_superquantile(z, alpha=0.95),
        num_samples=int(5e2),
        guide=inferred_parameters,
        method=method,
    )
    sq_optimal_prediction = RISK.propagate_uncertainty(opt_results.x)
    qois_sq = RISK.qoi(sq_optimal_prediction)
    sq_est = round_up(RISK.risk_measure(qois_sq))
    ouu_results = {
        "policy": opt_results.x,
        "risk": [sq_est],
        "samples": sq_optimal_prediction,
        "qoi": qois_sq,
        "tspan": RISK.tspan,
        "OptResults": opt_results,
    }
    print("Estimated risk at optimal policy", ouu_results["risk"])
    return ouu_results
