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
from pyciemss.risk.risk_measures import alpha_superquantile
import pyciemss.risk.qoi
from pyciemss.utils.interface_utils import convert_to_output_format, csv_to_list
from pyciemss.visuals import plots

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

from pyciemss.custom_decorators import pyciemss_logging_wrappper

# TODO: These interfaces should probably be just in terms of JSON-like objects.

PetriSolution = dict[str, torch.Tensor]
PetriInferredParameters = pyro.nn.PyroModule


@pyciemss_logging_wrappper
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
    compile_rate_law_p: bool = True,
    time_unit: Optional[str] = None,
    visual_options: Union[None, bool, dict[str, any]] = None,
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
        time_unit: str
            - Time unit (used for labeling outputs)
        visual_options: None, bool, dict[str, any]
            - True output a visual
            - False do not output a visual
            - dict output a visual with the dictionary passed to the visualization as kwargs

    Returns:
        samples:
            - PetriSolution: The samples from the model as a pandas DataFrame. (If visual_options is falsy)
            - dict {data: <samples>, visual: <visual>}: The PetriSolution and a visualization. (If visual_options is truthy)
    """
    model = load_petri_model(
        petri_model_or_path=petri_model_or_path,
        add_uncertainty=True,
        pseudocount=pseudocount,
        compile_rate_law_p=compile_rate_law_p,
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

    processed_samples = convert_to_output_format(
        samples, timepoints, interventions=interventions, time_unit=time_unit
    )

    if visual_options:
        visual_options = {} if visual_options is True else visual_options
        schema = plots.trajectories(processed_samples, **visual_options)
        return {"data": processed_samples, "visual": schema}
    else:
        return processed_samples


@pyciemss_logging_wrappper
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
    compile_rate_law_p: bool = True,
    time_unit: Optional[str] = None,
    visual_options: Union[None, bool, dict[str, any]] = None,
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
        time_unit: str
            - Time unit (used for labeling outputs)
        visual_options: None, bool, dict[str, any]
            - True output a visual
            - False do not output a visual
            - dict output a visual with the dictionary passed to the visualization as kwargs

    Returns:
        samples:
            - PetriSolution: The samples from the model as a pandas DataFrame. (If visual_options is falsy)
            - dict {data: <samples>, visual: <visual>}: The PetriSolution and a visualization. (If visual_options is truthy)
    """
    data = csv_to_list(data_path)

    model = load_petri_model(
        petri_model_or_path=petri_model_or_path,
        add_uncertainty=True,
        compile_rate_law_p=compile_rate_law_p,
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

    processed_samples = convert_to_output_format(
        samples, timepoints, interventions=interventions, time_unit=time_unit
    )

    if visual_options:
        visual_options = {} if visual_options is True else visual_options
        schema = plots.trajectories(processed_samples, **visual_options)
        return {"data": processed_samples, "visual": schema}
    else:
        return processed_samples


def load_and_optimize_and_sample_petri_model(
    petri_model_or_path: Union[str, mira.metamodel.TemplateModel, mira.modeling.Model],
    num_samples: int,
    timepoints: Iterable[float],
    interventions: Iterable[Tuple[float, str]],
    qoi: Iterable[Tuple[str, str, float]],
    risk_bound: float,
    objfun: callable = lambda x: np.abs(x),
    initial_guess: Iterable[float] = 0.5,
    bounds: Iterable[float] = [[0.0], [1.0]],
    *,
    start_state: Optional[dict[str, float]] = None,
    pseudocount: float = 1.0,
    start_time: float = -1e-10,
    method="dopri5",
    verbose: bool = False,
    n_samples_ouu: int = int(1e2),
    maxiter: int = 2,
    maxfeval: int = 25,
) -> Tuple[pd.DataFrame, dict]:
    """
    Load a petri net from a file, compile it into a probabilistic program, optimize under uncertainty,
    sample for the optimal intervention, and estinate risk.

    Args:
        petri_model_or_path: Union[str, mira.metamodel.TemplateModel, mira.modeling.Model]
            - A path to a petri net file, or a petri net object.
            - This path can be a URL or a local path to a mira model or AMR model.
            - Alternatively, this can be a mira template model directly.
        num_samples: int
            - The number of samples to draw from the model.
        timepoints: [Iterable[float]]
            - The timepoints to simulate the model from. Backcasting and/or forecasting is reflected in the choice of timepoints.
        interventions: Iterable[Tuple[float, str]]
            - A list of interventions to apply to the model. Each intervention is a tuple of the form (time, parameter_name).
        qoi: Tuple[str, str, **args]
            - The quantity of interest to optimize over. QoI is a tuple of the form (callable_qoi_function_name, state_variable_name, function_arguments).
            - Options for "callable_qoi_function_name":
                - scenario2dec_nday_average: performs average over last ndays of timepoints
        risk_bound: float
            - The threshold on the risk constraint.
        objfun: callable
            - The objective function defined as a callable function definition. E.g., to minimize the absolute value of intervention parameters use lambda x: np.sum(np.abs(x))
        initial_guess: Iterable[float]
            - The initial guess for the optimizer
        bounds: Iterable[float]
            - The lower and upper bounds for intervention parameter. Bounds are a list of the form [[lower bounds], [upper bounds]]
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
        verbose: bool
            - Whether to print out the optimization under uncertainty progress.
        n_samples_ouu: int
            - The number of samples to draw from the model for each optimization iteration.
        maxiter: int >= 0
            - The maximum number of restarts for multi-start local optimization.
            - maxiter = 0: leads to a single-start local optimization
        maxfeval: int > 0
            - The maximum number of function evaluations for each start of the local optimizer.

    Returns:
        samples: pd.DataFrame
            - The samples from the model using the optimal policy under uncertainty returned as a pandas DataFrame.
        optimal_policy: dict
            - Optimal policy under uncertainty returned as a dictionary with the following attributes:
                * policy: Optimal intervention
                * OptResults: Optimization results as scipy optimization object
                * risk: Estimated alpha-superquantile risk with alpha=0.95
                * samples: Samples from the model at the optimal intervention
                * qoi: Samples of quantity of interest
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

    def qoi_fn(y):
        return getattr(pyciemss.risk.qoi, qoi[0])(y, [qoi[1]], *qoi[2:])

    ouu_policy = optimize(
        model,
        timepoints=timepoints,
        interventions=interventions,
        qoi=qoi_fn,
        risk_bound=risk_bound,
        objfun=objfun,
        initial_guess=initial_guess,
        bounds=bounds,
        n_samples_ouu=n_samples_ouu,
        maxiter=maxiter,
        maxfeval=maxfeval,
        method=method,
        verbose=verbose,
        postprocess=False,
    )

    # Post-process OUU results
    if verbose:
        print("Post-processing optimal policy...")
    control_model = copy.deepcopy(model)
    RISK = computeRisk(
        model=control_model,
        interventions=interventions,
        qoi=qoi_fn,
        tspan=timepoints,
        risk_measure=lambda z: alpha_superquantile(z, alpha=0.95),
        num_samples=num_samples,
        method=method,
    )
    sq_optimal_prediction = RISK.propagate_uncertainty(ouu_policy["policy"])
    qois_sq = RISK.qoi(sq_optimal_prediction)
    sq_est = RISK.risk_measure(qois_sq)
    ouu_results = {"risk": [sq_est], "samples": sq_optimal_prediction, "qoi": qois_sq}
    ouu_policy.update(ouu_results)

    if verbose:
        print("Estimated risk at optimal policy", ouu_policy["risk"])

    x = list(np.atleast_1d(ouu_policy["policy"]))
    interventions_opt = []
    for intervention, value in zip(interventions, x):
        interventions_opt.append((intervention[0], intervention[1], value))

    samples = ouu_policy["samples"]

    processed_samples = convert_to_output_format(
        samples, timepoints, interventions=interventions_opt
    )

    return processed_samples, ouu_policy


def load_and_calibrate_and_optimize_and_sample_petri_model(
    petri_model_or_path: Union[str, mira.metamodel.TemplateModel, mira.modeling.Model],
    data_path: str,
    num_samples: int,
    timepoints: Iterable[float],
    interventions: Iterable[Tuple[float, str]],
    qoi: Iterable[Tuple[str, str, float]],
    risk_bound: float,
    objfun: callable = lambda x: np.abs(x),
    initial_guess: Iterable[float] = 0.5,
    bounds: Iterable[float] = [[0.0], [1.0]],
    *,
    start_state: Optional[dict[str, float]] = None,
    pseudocount: float = 1.0,
    start_time: float = -1e-10,
    num_iterations: int = 1000,
    lr: float = 0.03,
    num_particles: int = 1,
    autoguide=pyro.infer.autoguide.AutoLowRankMultivariateNormal,
    method="dopri5",
    verbose: bool = False,
    n_samples_ouu: int = int(1e2),
    maxiter: int = 2,
    maxfeval: int = 25,
) -> Tuple[pd.DataFrame, dict]:
    """
    Load a petri net from a file, compile it into a probabilistic program, calibrate on data, optimize under uncertainty,
    sample for the optimal policy, and estinate risk for the optimal policy.

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
        interventions: Iterable[Tuple[float, str]]
            - A list of interventions to apply to the model. Each intervention is a tuple of the form (time, parameter_name).
        qoi: Tuple[str, str, **args]
            - Quantity of interest to optimize over. QoI is a tuple of the form (callable_qoi_function_name, state_variable_name, function_arguments).
            - Options for "callable_qoi_function_name":
                - scenario2dec_nday_average: performs average over last ndays of timepoints
        risk_bound: float
            - Bound on the risk constraint.
        objfun: callable
            - Objective function as a callable function definition. E.g., to minimize the absolute value of intervention parameters use lambda x: np.sum(np.abs(x))
        initial_guess: Iterable[float]
            - Initial guess for the optimizer
        bounds: Iterable[float]
            - Lower and upper bounds for intervention parameter. Bounds are a list of the form [[lower bounds], [upper bounds]]
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
            - Whether to print out the calibration progress and the optimization under uncertainty progress. This will include summaries of the evidence lower bound (ELBO) and the parameters.
        num_particles: int > 0
            - The number of particles to use for the calibration. Increasing this value will result in lower variance gradient estimates, but will also increase the computational cost per gradient step.
        autoguide: pyro.infer.autoguide.AutoGuide
            - The guide to use for the calibration. By default we use the AutoLowRankMultivariateNormal guide. This is an advanced option. Please see the Pyro documentation for more details.
        method: str
            - The method to use for solving the ODE. See torchdiffeq's `odeint` method for more details.
            - If performance is incredibly slow, we suggest using `euler` to debug. If using `euler` results in faster simulation, the issue is likely that the model is stiff.
        n_samples_ouu: int
            - The number of samples to draw from the model for each optimization iteration.
        maxiter: int >= 0
            - The maximum number of restarts for multi-start local optimization.
            - maxiter = 0: leads to a single-start local optimization
        maxfeval: int > 0
            - The maximum number of function evaluations for each start of the local optimizer.


    Returns:
        samples: pd.DataFrame
            - The samples from the model using the optimal policy under uncertainty after calibrating on given data returned as a pandas DataFrame.
        optimal_policy: dict
            - Optimal policy under uncertainty returned as a dictionary with the following attributes:
                * policy: Optimal intervention
                * OptResults: Optimization results as scipy optimization object
                * risk: Estimated alpha-superquantile risk with alpha=0.95
                * samples: Samples from the model at the optimal intervention
                * qoi: Samples of quantity of interest
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

    def qoi_fn(y):
        return getattr(pyciemss.risk.qoi, qoi[0])(y, [qoi[1]], *qoi[2:])

    ouu_policy = optimize(
        model,
        timepoints=timepoints,
        interventions=interventions,
        qoi=qoi_fn,
        risk_bound=risk_bound,
        objfun=objfun,
        initial_guess=initial_guess,
        bounds=bounds,
        inferred_parameters=inferred_parameters,
        n_samples_ouu=n_samples_ouu,
        maxiter=maxiter,
        maxfeval=maxfeval,
        method=method,
        verbose=verbose,
        postprocess=False,
    )

    # Post-process OUU results
    if verbose:
        print("Post-processing optimal policy...")
    control_model = copy.deepcopy(model)
    RISK = computeRisk(
        model=control_model,
        interventions=interventions,
        qoi=qoi_fn,
        tspan=timepoints,
        risk_measure=lambda z: alpha_superquantile(z, alpha=0.95),
        num_samples=num_samples,
        method=method,
        guide=inferred_parameters,
    )
    sq_optimal_prediction = RISK.propagate_uncertainty(ouu_policy["policy"])
    qois_sq = RISK.qoi(sq_optimal_prediction)
    sq_est = RISK.risk_measure(qois_sq)
    ouu_results = {"risk": [sq_est], "samples": sq_optimal_prediction, "qoi": qois_sq}
    ouu_policy.update(ouu_results)

    if verbose:
        print("Estimated risk at optimal policy", ouu_policy["risk"])

    x = list(np.atleast_1d(ouu_policy["policy"]))
    interventions_opt = []
    for intervention, value in zip(interventions, x):
        interventions_opt.append((intervention[0], intervention[1], value))

    samples = ouu_policy["samples"]

    processed_samples = convert_to_output_format(
        samples, timepoints, interventions=interventions_opt
    )

    return processed_samples, ouu_policy


##############################################################################
# Internal Interfaces Below - TA4 above


def load_petri_model(
    petri_model_or_path: Union[str, mira.metamodel.TemplateModel, mira.modeling.Model],
    add_uncertainty=True,
    pseudocount=1.0,
    compile_rate_law_p: bool = False,
) -> PetriNetODESystem:
    """
    Load a petri net from a file and compile it into a probabilistic program.
    """
    if add_uncertainty:
        model = ScaledBetaNoisePetriNetODESystem.from_askenet(
            petri_model_or_path, compile_rate_law_p=compile_rate_law_p
        )
        model.pseudocount = torch.tensor(pseudocount)
        return model
    else:
        return MiraPetriNetODESystem.from_askenet(
            petri_model_or_path, compile_rate_law_p=compile_rate_law_p
        )


@setup_model.register
@pyciemss_logging_wrappper
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
@pyciemss_logging_wrappper
def reset_petri_model(petri: PetriNetODESystem) -> PetriNetODESystem:
    """
    Reset a model to its initial state.
    reset_model * setup_model = id
    """
    new_petri = copy.deepcopy(petri)
    new_petri.reset()
    return new_petri


@intervene.register
@pyciemss_logging_wrappper
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
@pyciemss_logging_wrappper
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
@pyciemss_logging_wrappper
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
@pyciemss_logging_wrappper
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
    verbose: bool = False,
    postprocess: bool = True,
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
    if verbose:
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
    if verbose:
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
    if verbose:
        print(f"Optimization completed in time {time.time()-start_time:.2f} seconds.")
        print(f"Optimal policy:\t{opt_results.x}")

    # Check for some interventions that lead to no feasible solutions
    if opt_results.x < 0:
        if verbose:
            print("No solution found")

    # Post-process OUU results
    if postprocess:
        if verbose:
            print("Post-processing optimal policy...")
        # TODO: check best way to set tspan for plotting
        tspan_plot = [float(x) for x in list(range(0, int(timepoints[-1])))]
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
        if verbose:
            print("Estimated risk at optimal policy", ouu_results["risk"])
    else:
        ouu_results = {
            "policy": opt_results.x,
            "OptResults": opt_results,
        }
    return ouu_results
