import logging
import os
import json
import pyro
import torch
import time
import numpy as np
from math import ceil
import pandas as pd
from typing import Iterable, Optional, Tuple, Union, Callable
import copy
import warnings

import random as rand

from pyro.infer import Predictive
from pyro.infer.autoguide import AutoDelta, AutoLowRankMultivariateNormal, AutoGuideList

from pyciemss.PetriNetODE.base import (
    PetriNetODESystem,
    ScaledNormalNoisePetriNetODESystem,
    ScaledBetaNoisePetriNetODESystem,
    MiraPetriNetODESystem,
    get_name,
)
from pyciemss.risk.ouu import computeRisk, solveOUU
from pyciemss.risk.risk_measures import alpha_superquantile
import pyciemss.risk.qoi
from pyciemss.utils.interface_utils import convert_to_output_format, csv_to_list
from pyciemss.visuals import plots

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

from pyciemss.custom_decorators import pyciemss_logging_wrapper

# TODO: These interfaces should probably be just in terms of JSON-like objects.

PetriSolution = dict[str, torch.Tensor]
PetriInferredParameters = pyro.nn.PyroModule

@pyciemss_logging_wrapper
def load_and_sample_petri_model(
    petri_model_or_path: Union[str, mira.metamodel.TemplateModel, mira.modeling.Model],
    num_samples: int,
    timepoints: Iterable[float],
    *,
    interventions: Optional[Iterable[Tuple[float, str, float]]] = None,
    start_state: Optional[dict[str, float]] = None,
    start_time: float = -1e-10,
    method="dopri5",
    compile_rate_law_p: bool = True,
    compile_observables_p = True,
    time_unit: Optional[str] = None,
    visual_options: Union[None, bool, dict[str, any]] = None,
    alpha_qs: Optional[Iterable[float]] = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99],
    stacking_order: Optional[str] = "timepoints",
) -> dict:
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
        start_time: float
            - The start time of the model. This is used to align the `start_state` with the `timepoints`.
            - By default we set the `start_time` to be a small negative number to avoid numerical issues w/ collision with the `timepoints` which typically start at 0.
        method: str
            - The method to use for solving the ODE. See torchdiffeq's `odeint` method for more details.
            - If performance is incredibly slow, we suggest using `euler` to debug. If using `euler` results in faster simulation, the issue is likely that the model is stiff.
        compile_rate_law_p: bool
            - Whether or not to compile the rate law from the AMR rate expression. default True.  False is useful for debugging.
        compile_observables_p: bool
            - Whether or not to compile the observables from the AMR observable expression. default True.  False is useful for debugging.
        time_unit: str
            - Time unit (used for labeling outputs)
        visual_options: None, bool, dict[str, any]
            - True output a visual
            - False do not output a visual
            - dict output a visual with the dictionary passed to the visualization as kwargs
        alpha_qs: Optional[Iterable[float]]
            - The quantiles required for estimating weighted interval score to test forecasting accuracy.
        stacking_order: Optional[str]
            - The stacking order requested for the quantiles to keep the selected quantity together for each state.
            - Options: "timepoints" or "quantiles"

    Returns:
        result: dict
            - Dictionary of outputs with following attribute:
                * data: The samples from the model as a pandas DataFrame.
                * quantiles: The quantiles for ensemble score calculation as a pandas DataFrames.
                * state: Risk estimates for each state as 2-day average at the final timepoint
                    * risk: Estimated alpha-superquantile risk with alpha=0.95
                    * qoi: Samples of quantity of interest (in this case, 2-day average of the state at the final timepoint)
                * visual: Visualization. (If visual_options is truthy)
    """

    # Load the model
    model = load_petri_model(
        petri_model_or_path=petri_model_or_path,
        add_uncertainty=False,
        compile_rate_law_p=compile_rate_law_p,
        compile_observables_p=compile_observables_p,
    )

    # If the user doesn't override the start state, use the initial values from the model.
    if start_state is None:
        start_state = {
            get_name(v): v.data["initial_value"] for v in model.G.variables.values()
        }

    model = setup_model(model, start_time=start_time, start_state=start_state)

    if interventions is not None:
        model = intervene(model, interventions)

    samples = sample(
        model,
        timepoints,
        num_samples,
        method=method,
    )
    
    def qoi_fn(y):
        return getattr(pyciemss.risk.qoi, qoi[0])(y, [qoi[1]], *qoi[2:])    
    risk_results = {}
    for k, vals in samples.items():
        if "_sol" in k:
            qoi = ("scenario2dec_nday_average", k, 2)
            qois_sq = qoi_fn(samples)
            sq_est = alpha_superquantile(qois_sq, alpha=0.95)
            risk_results.update({k: {"risk": [sq_est], "qoi": qois_sq}})

    if compile_observables_p:
        observables = model.compiled_observables
    else:
        observables = None
    
    processed_samples, q_ensemble = convert_to_output_format(
        samples, timepoints, interventions=interventions, time_unit=time_unit,
        quantiles=True, alpha_qs=alpha_qs, stacking_order=stacking_order,
        observables=observables
    )

    if visual_options:
        visual_options = {} if visual_options is True else visual_options
        schema = plots.trajectories(processed_samples, **visual_options)
        return {"data": processed_samples, "quantiles": q_ensemble, "risk": risk_results, "visual": schema}
    else:
        return {"data": processed_samples, "quantiles": q_ensemble, "risk": risk_results}


@pyciemss_logging_wrapper
def load_and_calibrate_and_sample_petri_model(
    petri_model_or_path: Union[str, mira.metamodel.TemplateModel, mira.modeling.Model],
    data_path: str,
    num_samples: int,
    timepoints: Iterable[float],
    *,
    noise_model: str = "scaled_normal",
    noise_scale: float = 0.1,
    interventions: Optional[Iterable[Tuple[float, str, float]]] = None,
    start_state: Optional[dict[str, float]] = None,
    start_time: float = -1e-10,
    num_iterations: int = 1000,
    lr: float = 0.03,
    verbose: bool = False,
    num_particles: int = 1,
    deterministic_learnable_parameters: Iterable[str] = [],
    method="dopri5",
    compile_rate_law_p: bool = True,
    compile_observables_p = True,
    time_unit: Optional[str] = None,
    visual_options: Union[None, bool, dict[str, any]] = None,
    progress_hook: Callable = lambda _: None,
    alpha_qs: Optional[Iterable[float]] = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99],
    stacking_order: Optional[str] = "timepoints",
) -> dict:
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
        noise_model: str
            - The noise model to use for the model.
            - Options are "scaled_beta" and "scaled_normal".
        noise_scale: float > 0.0
            - A scaling parameter for the noise model.
            - Larger values of noise_scale correspond to more certainty about the model parameters.
        interventions: Optional[Iterable[Tuple[float, str, float]]]
            - A list of interventions to apply to the model. Each intervention is a tuple of the form (time, parameter_name, value).
        start_state: Optional[dict[str, float]]
            - The initial state of the model. If None, the initial state is taken from the mira model.
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
        compile_rate_law_p: bool
            - Whether or not to compile the rate law from the AMR rate expression. default True.  False is useful for debugging.
        compile_observables_p: bool
            - Whether or not to compile the observables from the AMR observable expression. default True.  False is useful for debugging.
        time_unit: str
            - Time unit (used for labeling outputs)
        visual_options: None, bool, dict[str, any]
            - True output a visual
            - False do not output a visual
            - dict output a visual with the dictionary passed to the visualization as kwargs
        progress_hook: Callable
            - The hook transmitting the current progress of the calibration.
        alpha_qs: Optional[Iterable[float]]
            - The quantiles required for estimating weighted interval score to test ensemble forecasting accuracy.
        stacking_order: Optional[str]
            - The stacking order requested for the ensemble quantiles to keep the selected quantity together for each state.
            - Options: "timepoints" or "quantiles"

    Returns:
        result: dict
            - Dictionary of outputs with following attribute:
                * data: The samples from the calibrated model as a pandas DataFrame.
                * quantiles: The quantiles for ensemble score calculation after calibration as a pandas DataFrames.
                * state: Risk estimates for each state as 2-day average at the final timepoint
                    * risk: Estimated alpha-superquantile risk with alpha=0.95
                    * qoi: Samples of quantity of interest (in this case, 2-day average of the state at the final timepoint)
                * visual: Visualization. (If visual_options is truthy)
    """
    data = csv_to_list(data_path)

    model = load_petri_model(
        petri_model_or_path=petri_model_or_path,
        add_uncertainty=False,
        noise_model=noise_model,
        noise_scale=noise_scale,
        compile_rate_law_p=compile_rate_law_p,
        compile_observables_p=compile_observables_p,
    )

    # If the user doesn't override the start state, use the initial values from the model.
    if start_state is None:
        start_state = {
            get_name(v): v.data["initial_value"] for v in model.G.variables.values()
        }

    model = setup_model(model, start_time=start_time, start_state=start_state)

    if interventions is not None:
        model = intervene(model, interventions)

    def autoguide(model):
        guide = AutoGuideList(model)
        guide.append(AutoDelta(pyro.poutine.block(model, expose=deterministic_learnable_parameters)))
        guide.append(AutoLowRankMultivariateNormal(pyro.poutine.block(model, hide=deterministic_learnable_parameters)))
        return guide

    inferred_parameters = calibrate(
        model,
        data,
        num_iterations,
        lr,
        verbose,
        num_particles,
        autoguide,
        method=method,
        progress_hook=progress_hook
    )
    samples = sample(
        model,
        timepoints,
        num_samples,
        inferred_parameters=inferred_parameters,
        method=method,
    )

    def qoi_fn(y):
        return getattr(pyciemss.risk.qoi, qoi[0])(y, [qoi[1]], *qoi[2:])    
    risk_results = {}
    for k, vals in samples.items():
        if "_sol" in k:
            qoi = ("scenario2dec_nday_average", k, 2)
            qois_sq = qoi_fn(samples)
            sq_est = alpha_superquantile(qois_sq, alpha=0.95)
            risk_results.update({k: {"risk": [sq_est], "qoi": qois_sq}})

    if compile_observables_p:
        observables = model.compiled_observables
    else:
        observables = None

    processed_samples, q_ensemble = convert_to_output_format(
        samples, timepoints, interventions=interventions, time_unit=time_unit,
        quantiles=True, alpha_qs=alpha_qs, stacking_order=stacking_order,
        observables=observables,
        train_end_point = max([d[0] for d in data])
    )

    if visual_options:
        visual_options = {} if visual_options is True else visual_options
        schema = plots.trajectories(processed_samples, **visual_options)
        return {"data": processed_samples, "quantiles": q_ensemble, "risk": risk_results, "visual": schema}
    else:
        return {"data": processed_samples, "quantiles": q_ensemble, "risk": risk_results}

@pyciemss_logging_wrapper
def load_and_optimize_and_sample_petri_model(
    petri_model_or_path: Union[str, mira.metamodel.TemplateModel, mira.modeling.Model],
    num_samples: int,
    timepoints: Iterable[float],
    interventions: Iterable[Tuple[float, str]],
    qoi: Tuple[str, str, float] = ("scenario2dec_nday_average", "I_sol", 2),
    risk_bound: float = 1.0,
    objfun: callable = lambda x: np.sum(np.abs(x)),
    initial_guess: Iterable[float] = 0.5,
    bounds: Iterable[float] = [[0.0], [1.0]],
    *,
    start_state: Optional[dict[str, float]] = None,
    start_time: float = -1e-10,
    method="dopri5",
    compile_rate_law_p: bool = True,
    compile_observables_p: bool = True,
    verbose: bool = False,
    n_samples_ouu: int = int(1e2),
    maxiter: int = 2,
    maxfeval: int = 25,
    time_unit: Optional[str] = None,
    visual_options: Union[None, bool, dict[str, any]] = None,
    alpha_qs: Optional[Iterable[float]] = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99],
    stacking_order: Optional[str] = "timepoints",
) -> dict:
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
        start_time: float
            - The start time of the model. This is used to align the `start_state` with the `timepoints`.
            - By default we set the `start_time` to be a small negative number to avoid numerical issues w/ collision with the `timepoints` which typically start at 0.
        method: str
            - The method to use for solving the ODE. See torchdiffeq's `odeint` method for more details.
            - If performance is incredibly slow, we suggest using `euler` to debug. If using `euler` results in faster simulation, the issue is likely that the model is stiff.
        compile_rate_law_p: bool
            - Whether or not to compile the rate law from the AMR rate expression. default True.  False is useful for debugging.
        compile_observables_p: bool
            - Whether or not to compile the observables from the AMR observable expression. default True.  False is useful for debugging.
        verbose: bool
            - Whether to print out the optimization under uncertainty progress.
        n_samples_ouu: int
            - The number of samples to draw from the model for each optimization iteration.
        maxiter: int >= 0
            - The maximum number of restarts for multi-start local optimization.
            - maxiter = 0: leads to a single-start local optimization
        maxfeval: int > 0
            - The maximum number of function evaluations for each start of the local optimizer.
        time_unit: str
            - Time unit (used for labeling outputs)
        visual_options: None, bool, dict[str, any]
            - True output a visual
            - False do not output a visual
            - dict output a visual with the dictionary passed to the visualization as kwargs
        alpha_qs: Optional[Iterable[float]]
            - The quantiles required for estimating weighted interval score to test ensemble forecasting accuracy.
        stacking_order: Optional[str]
            - The stacking order requested for the ensemble quantiles to keep the selected quantity together for each state.
            - Options: "timepoints" or "quantiles"

    Returns:
        result: dict
            - Dictionary of outputs with following attribute:
                * data: pd.DataFrame
                    - The samples from the model using the optimal policy under uncertainty returned as a pandas DataFrame.
                * policy: dict
                    - Optimal policy under uncertainty returned as a dictionary with the following attributes:
                        * policy: Optimal intervention
                        * OptResults: Optimization results as scipy optimization object
                        * risk: Estimated alpha-superquantile risk with alpha=0.95
                        * samples: Samples from the model at the optimal intervention
                        * qoi: Samples of quantity of interest
                * quantiles: The quantiles for ensemble score calculation after calibration as a pandas DataFrames.
                * visual: Visualization. (If visual_options is truthy)
    """
    model = load_petri_model(
        petri_model_or_path=petri_model_or_path,
        add_uncertainty=False,
        compile_rate_law_p=compile_rate_law_p,
        compile_observables_p=compile_observables_p,
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

    if compile_observables_p:
        observables = model.compiled_observables
    else:
        observables = None

    processed_samples, q_ensemble = convert_to_output_format(
        samples, timepoints, interventions=interventions_opt, time_unit=time_unit,
        quantiles=True, alpha_qs=alpha_qs, stacking_order=stacking_order,
        observables=observables
    )

    if visual_options:
        visual_options = {} if visual_options is True else visual_options
        schema = plots.trajectories(processed_samples, **visual_options)
        return {"data": processed_samples, "policy": ouu_policy, "quantiles": q_ensemble, "visual": schema}
    else:
        return {"data": processed_samples, "policy": ouu_policy, "quantiles": q_ensemble}

@pyciemss_logging_wrapper
def load_and_calibrate_and_optimize_and_sample_petri_model(
    petri_model_or_path: Union[str, mira.metamodel.TemplateModel, mira.modeling.Model],
    data_path: str,
    num_samples: int,
    timepoints: Iterable[float],
    interventions: Iterable[Tuple[float, str]],
    qoi: Tuple[str, str, float] = ("scenario2dec_nday_average", "I_sol", 2),
    risk_bound: float = 1.0,
    objfun: callable = lambda x: np.sum(np.abs(x)),
    initial_guess: Iterable[float] = 0.5,
    bounds: Iterable[float] = [[0.0], [1.0]],
    *,
    noise_model: str = "scaled_normal",
    noise_scale: float = 0.1,
    start_state: Optional[dict[str, float]] = None,
    start_time: float = -1e-10,
    num_iterations: int = 1000,
    lr: float = 0.03,
    num_particles: int = 1,
    deterministic_learnable_parameters: Iterable[str] = [],
    method="dopri5",
    verbose: bool = False,
    n_samples_ouu: int = int(1e2),
    compile_rate_law_p: bool = True,
    compile_observables_p: bool = True,
    maxiter: int = 2,
    maxfeval: int = 25,
    time_unit: Optional[str] = None,
    visual_options: Union[None, bool, dict[str, any]] = None,
    progress_hook: Callable = lambda _: None,
    alpha_qs: Optional[Iterable[float]] = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99],
    stacking_order: Optional[str] = "timepoints",
) -> dict:
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
            - The pseudocount to use for adding uncertainty to the observations.
            - Larger values of pseudocount correspond to more certainty about the observations.
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
        compile_rate_law_p: bool
            - Whether or not to compile the rate law from the AMR rate expression. default True.  False is useful for debugging.
        compile_observables_p: bool
            - Whether or not to compile the observables from the AMR observable expression. default True.  False is useful for debugging.    
        n_samples_ouu: int
            - The number of samples to draw from the model for each optimization iteration.
        maxiter: int >= 0
            - The maximum number of restarts for multi-start local optimization.
            - maxiter = 0: leads to a single-start local optimization
        maxfeval: int > 0
            - The maximum number of function evaluations for each start of the local optimizer.
        time_unit: str
            - Time unit (used for labeling outputs)
        visual_options: None, bool, dict[str, any]
            - True output a visual
            - False do not output a visual
            - dict output a visual with the dictionary passed to the visualization as kwargs
        progress_hook: Callable
            - The hook transmitting the current progress of the calibration.
        alpha_qs: Optional[Iterable[float]]
            - The quantiles required for estimating weighted interval score to test ensemble forecasting accuracy.
        stacking_order: Optional[str]
            - The stacking order requested for the ensemble quantiles to keep the selected quantity together for each state.
            - Options: "timepoints" or "quantiles"

    Returns:
        result: dict
            - Dictionary of outputs with following attribute:
                * data: pd.DataFrame
                    - The samples from the model using the optimal policy under uncertainty after calibrating on given data returned as a pandas DataFrame.
                * policy: dict
                    - Optimal policy under uncertainty returned as a dictionary with the following attributes:
                        * policy: Optimal intervention
                        * OptResults: Optimization results as scipy optimization object
                        * risk: Estimated alpha-superquantile risk with alpha=0.95
                        * samples: Samples from the model at the optimal intervention
                        * qoi: Samples of quantity of interest
                * quantiles: The quantiles for ensemble score calculation after calibration as a pandas DataFrames.
                * visual: Visualization. (If visual_options is truthy)
    """
    data = csv_to_list(data_path)

    model = load_petri_model(
        petri_model_or_path=petri_model_or_path,
        add_uncertainty=False,
        noise_model=noise_model,
        noise_scale=noise_scale,
        compile_rate_law_p=compile_rate_law_p,
        compile_observables_p=compile_observables_p,
    )

    # If the user doesn't override the start state, use the initial values from the model.
    if start_state is None:
        start_state = {
            get_name(v): v.data["initial_value"] for v in model.G.variables.values()
        }

    model = setup_model(model, start_time=start_time, start_state=start_state)

    def autoguide(model):
        guide = AutoGuideList(model)
        guide.append(AutoDelta(pyro.poutine.block(model, expose=deterministic_learnable_parameters)))
        guide.append(AutoLowRankMultivariateNormal(pyro.poutine.block(model, hide=deterministic_learnable_parameters)))
        return guide

    inferred_parameters = calibrate(
        model,
        data,
        num_iterations,
        lr,
        verbose,
        num_particles,
        autoguide,
        method=method,
        progress_hook=progress_hook
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

    observables = None
    if compile_observables_p:
        observables = model.compiled_observables

    
    processed_samples, q_ensemble = convert_to_output_format(
        samples, timepoints, interventions=interventions_opt, time_unit=time_unit,
        quantiles=True, alpha_qs=alpha_qs, stacking_order=stacking_order,
        observables=observables,
        train_end_point = max([d[0] for d in data])
    )

    if visual_options:
        visual_options = {} if visual_options is True else visual_options
        schema = plots.trajectories(processed_samples, **visual_options)
        return {"data": processed_samples, "policy": ouu_policy, "quantiles": q_ensemble, "visual": schema}
    else:
        return {"data": processed_samples, "policy": ouu_policy, "quantiles": q_ensemble}


##############################################################################
# Internal Interfaces Below - TA4 above

def load_petri_model(
    petri_model_or_path: Union[str, mira.metamodel.TemplateModel, mira.modeling.Model],
    add_uncertainty: bool = True,
    noise_model: str = "scaled_normal",
    noise_scale: float = 0.1,
    compile_observables_p = False,
    compile_rate_law_p: bool = False) -> PetriNetODESystem:
    """
    Load a petri net from a file and compile it into a probabilistic program.
    """
    if noise_model == "scaled_beta":
        return ScaledBetaNoisePetriNetODESystem.from_askenet(
            petri_model_or_path, noise_scale=noise_scale, compile_rate_law_p=compile_rate_law_p, compile_observables_p=compile_observables_p, add_uncertainty=add_uncertainty
        )
    elif noise_model == "scaled_normal":
        return ScaledNormalNoisePetriNetODESystem.from_askenet(
            petri_model_or_path, noise_scale=noise_scale, compile_rate_law_p=compile_rate_law_p, compile_observables_p=compile_observables_p, add_uncertainty=add_uncertainty
        )
    else:
        raise ValueError(f"Unknown noise model {noise_model}. Please select from either 'scaled_beta' or 'scaled_normal'.")


@setup_model.register
@pyciemss_logging_wrapper
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
@pyciemss_logging_wrapper
def reset_petri_model(petri: PetriNetODESystem) -> PetriNetODESystem:
    """
    Reset a model to its initial state.
    reset_model * setup_model = id
    """
    new_petri = copy.deepcopy(petri)
    new_petri.reset()
    return new_petri


@intervene.register
@pyciemss_logging_wrapper
def intervene_petri_model(
    petri: PetriNetODESystem, interventions: Iterable[Tuple[float, str, float]], jostle_scale: float = 1e-5
) -> PetriNetODESystem:
    """
    Intervene on a model.
    """
    # Note: this will have to change if we want to add more sophisticated interventions.
    interventions = [
        StaticParameterInterventionEvent(timepoint + (0.1+rand.random())*jostle_scale, parameter, value)
        for timepoint, parameter, value in interventions
    ]
    new_petri = copy.deepcopy(petri)
    new_petri.load_events(interventions)
    return new_petri

@calibrate.register
@pyciemss_logging_wrapper
def calibrate_petri(
    petri: PetriNetODESystem,
    data: Iterable[Tuple[float, dict[str, float]]],
    num_iterations: int = 1000,
    lr: float = 0.03,
    verbose: bool = False,
    num_particles: int = 1,
    autoguide=pyro.infer.autoguide.AutoLowRankMultivariateNormal,
    method="dopri5",
    progress_hook: Callable = lambda _: None,
    jostle_scale: float = 1e-5,
) -> PetriInferredParameters:
    """
    Use variational inference with a mean-field variational family to infer the parameters of the model.
    """
    
    new_petri = copy.deepcopy(petri)
    observations = [
        ObservationEvent(timepoint + (0.1+rand.random()) * jostle_scale, observation) for timepoint, observation in data
    ]

    for obs in observations:
        s = 0.0
        for v in obs.observation.values():
            s += v
            if not 0 <= v <= petri.total_population:
                warnings.warn(f"Observation {obs} is not in the range [0, {petri.total_population}]. This may be an error!")
        #assert s <= petri.total_population or torch.isclose(s, petri.total_population)
    new_petri.load_events(observations)

    guide = autoguide(new_petri)
    optim = pyro.optim.Adam({"lr": lr})
    loss = pyro.infer.Trace_ELBO(num_particles=num_particles)
    svi = pyro.infer.SVI(new_petri, guide, optim, loss=loss)

    pyro.clear_param_store()

    for i in range(num_iterations):
        progress_hook(i/num_iterations)
        loss = svi.step(method=method)
        if verbose:
            if i % 25 == 0:
                print(f"iteration {i}: loss = {loss}")

    return guide


@sample.register
@pyciemss_logging_wrapper
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
@pyciemss_logging_wrapper
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
