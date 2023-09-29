from typing_extensions import deprecated
import pyro
import numpy as np
from typing import Iterable, Optional, Tuple, Union, Callable
import copy

from pyro.infer.autoguide import AutoDelta, AutoLowRankMultivariateNormal, AutoGuideList

from pyciemss.PetriNetODE.base import (
    get_name,
)
from pyciemss.risk.ouu import computeRisk
from pyciemss.risk.risk_measures import alpha_superquantile
import pyciemss.risk.qoi
from pyciemss.utils.interface_utils import csv_to_list

import mira


from pyciemss.custom_decorators import pyciemss_logging_wrapper

from .interfaces import (
    load_petri_model,
    setup_model,
    sample,
    calibrate,
    optimize,
    intervene,
    prepare_interchange_dictionary,
)
from ..interfaces import DEFAULT_QUANTILES


# TODO: These interfaces should probably be just in terms of JSON-like objects.


@deprecated(
    "'Big Box' interfaces were a stop-gap for July 2023 hackathon."
    "Please call the appropriate sequence from 'interfaces' insteads."
)
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
    compile_observables_p=True,
    time_unit: Optional[str] = None,
    visual_options: Union[None, bool, dict[str, any]] = None,
    alpha_qs: Optional[Iterable[float]] = DEFAULT_QUANTILES,
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
            - The timepoints to simulate the model from.
              Backcasting and/or forecasting is reflected in the choice of timepoints.
        interventions: Optional[Iterable[Tuple[float, str, float]]]
            - A list of interventions to apply to the model.
              Each intervention is a tuple of the form (time, parameter_name, value).
        start_state: Optional[dict[str, float]]
            - The initial state of the model. If None, the initial state is taken from the mira model.
        start_time: float
            - The start time of the model. This is used to align the `start_state` with the `timepoints`.
            - By default we set the `start_time` to be a small negative number to avoid numerical
              issues w/ collision with the `timepoints` which typically start at 0.
        method: str
            - The method to use for solving the ODE. See torchdiffeq's `odeint` method for more details.
            - If performance is incredibly slow, we suggest using `euler` to debug.
              If using `euler` results in faster simulation, the issue is likely that the model is stiff.
        compile_rate_law_p: bool
            - Whether or not to compile the rate law from the AMR rate expression. default True.
              False is useful for debugging.
        compile_observables_p: bool
            - Whether or not to compile the observables from the AMR observable expression. default True.
              False is useful for debugging.
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
                * risk: Risk estimates for each state as 2-day average at the final timepoint
                    * risk: Estimated alpha-superquantile risk with alpha=0.95
                    * qoi: Samples of quantity of interest
                          (in this case, 2-day average of the state at the final timepoint)
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

    result = prepare_interchange_dictionary(
        samples,
        timepoints,
        interventions=interventions,
        time_unit=time_unit,
        quantiles=True,
        alpha_qs=alpha_qs,
        stacking_order=stacking_order,
        observables=observables,
        visaul_options=visual_options,
    )

    result = {**result, "risk": risk_results}
    return result


@deprecated(
    "'Big Box' interfaces were a stop-gap for July 2023 hackathon."
    "Please call the appropriate sequence from 'interfaces' insteads."
)
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
    compile_observables_p=True,
    time_unit: Optional[str] = None,
    visual_options: Union[None, bool, dict[str, any]] = None,
    progress_hook: Callable = lambda _: None,
    alpha_qs: Optional[Iterable[float]] = DEFAULT_QUANTILES,
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
            - The path to the data to calibrate the model to.
              See notebook/integration_demo/data.csv for an example of the format.
        num_samples: int
            - The number of samples to draw from the model.
        timepoints: [Iterable[float]]
            - The timepoints to simulate the model from. Backcasting and/or forecasting is reflected
              in the choice of timepoints.
        noise_model: str
            - The noise model to use for the model.
            - Options are "scaled_beta" and "scaled_normal".
        noise_scale: float > 0.0
            - A scaling parameter for the noise model.
            - Larger values of noise_scale correspond to more certainty about the model parameters.
        interventions: Optional[Iterable[Tuple[float, str, float]]]
            - A list of interventions to apply to the model. Each intervention is a
              tuple of the form (time, parameter_name, value).
        start_state: Optional[dict[str, float]]
            - The initial state of the model. If None, the initial state is taken from the mira model.
        start_time: float
            - The start time of the model. This is used to align the `start_state` with the `timepoints`.
            - By default we set the `start_time` to be a small negative number to avoid numerical issues w/
              collision with the `timepoints` which typically start at 0.
        num_iterations: int > 0
            - The number of iterations to run the calibration for.
        lr: float > 0.0
            - The learning rate to use for the calibration.
        verbose: bool
            - Whether to print out the calibration progress. This will include summaries of
              the evidence lower bound (ELBO) and the parameters.
        num_particles: int > 0
            - The number of particles to use for the calibration. Increasing this value will result in lower
            variance gradient estimates, but will also increase the computational cost per gradient step.
        deterministic_learnable_parameters: Iterable[str]
            - The set of parameters whose calibration output will be point estimates that were learned from data.
        method: str
            - The method to use for the ODE solver. See `torchdiffeq.odeint` for more details.
            - If performance is incredibly slow, we suggest using `euler` to debug.
              If using `euler` results in faster simulation,
            the issue is likely that the model is stiff.
        compile_rate_law_p: bool
            - Whether or not to compile the rate law from the AMR rate expression. default True.
              False is useful for debugging.
        compile_observables_p: bool
            - Whether or not to compile the observables from the AMR observable expression. default True.
              False is useful for debugging.
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
            - The stacking order requested for the ensemble quantiles to keep the selected
              quantity together for each state.
            - Options: "timepoints" or "quantiles"

    Returns:
        result: dict
            - Dictionary of outputs with following attribute:
                * data: The samples from the calibrated model as a pandas DataFrame.
                * quantiles: The quantiles for ensemble score calculation after calibration as a pandas DataFrames.
                * risk: Risk estimates for each state as 2-day average at the final timepoint
                    * risk: Estimated alpha-superquantile risk with alpha=0.95
                    * qoi: Samples of quantity of interest (in this case, 2-day average of the
                           state at the final timepoint)
                * inferred_parameters: The inferred parameters from the calibration as PetriInferredParameters.
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
        guide.append(
            AutoDelta(
                pyro.poutine.block(model, expose=deterministic_learnable_parameters)
            )
        )
        guide.append(
            AutoLowRankMultivariateNormal(
                pyro.poutine.block(model, hide=deterministic_learnable_parameters)
            )
        )
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
        progress_hook=progress_hook,
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

    result = prepare_interchange_dictionary(
        samples,
        timepoints,
        interventions=interventions,
        time_unit=time_unit,
        quantiles=True,
        alpha_qs=alpha_qs,
        stacking_order=stacking_order,
        observables=observables,
        train_end_point=max([d[0] for d in data]),
        visual_options=visual_options,
    )
    result = {
        **result,
        "risk": risk_results,
        "inferred_parameters": inferred_parameters,
    }

    return result


@deprecated(
    "'Big Box' interfaces were a stop-gap for July 2023 hackathon."
    "Please call the appropriate sequence from 'interfaces' insteads."
)
@pyciemss_logging_wrapper
def load_and_optimize_and_sample_petri_model(
    petri_model_or_path: Union[str, mira.metamodel.TemplateModel, mira.modeling.Model],
    num_samples: int,
    timepoints: Iterable[float],
    interventions: Iterable[Tuple[float, str]],
    qoi: Tuple[str, str, float] = ("scenario2dec_nday_average", "I_sol", 2),
    risk_bound: float = 1.0,
    objfun: callable = lambda x: np.sum(np.abs(x)),
    initial_guess: Iterable[float] = [0.5],
    bounds: Iterable[Iterable[float]] = [[0.0], [1.0]],
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
    alpha_qs: Optional[Iterable[float]] = DEFAULT_QUANTILES,
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
            - The timepoints to simulate the model from. Backcasting and/or forecasting is
              reflected in the choice of timepoints.
        interventions: Iterable[Tuple[float, str]]
            - A list of interventions to apply to the model.
              Each intervention is a tuple of the form (time, parameter_name).
        qoi: Tuple[str, str, **args]
            - The quantity of interest to optimize over.
              QoI is a tuple of the form(callable_qoi_function_name, state_variable_name, function_arguments).
            - Options for "callable_qoi_function_name":
                - scenario2dec_nday_average: performs average over last ndays of timepoints
        risk_bound: float
            - The threshold on the risk constraint.
        objfun: callable
            - The objective function defined as a callable function definition.
              E.g., to minimize the absolute value of intervention parameters use lambda x: np.sum(np.abs(x))
        initial_guess: Iterable[float]
            - The initial guess for the optimizer.
              The length should be equal to number of dimensions of the intervention (or control action).
        bounds: Iterable[Iterable[float]]
            - The lower and upper bounds for intervention parameter.
              Bounds are a list of the form [[lower bounds], [upper bounds]]
        start_state: Optional[dict[str, float]]
            - The initial state of the model. If None, the initial state is taken from the mira model.
        start_time: float
            - The start time of the model. This is used to align the `start_state` with the `timepoints`.
            - By default we set the `start_time` to be a small negative number to avoid numerical
              issues w/collision with the `timepoints` which typically start at 0.
        method: str
            - The method to use for solving the ODE. See torchdiffeq's `odeint` method for more details.
            - If performance is incredibly slow, we suggest using `euler` to debug.
              If using `euler` results in faster simulation, the issue is likely that the model is stiff.
        compile_rate_law_p: bool
            - Whether or not to compile the rate law from the AMR rate expression. default True.
              False is useful for debugging.
        compile_observables_p: bool
            - Whether or not to compile the observables from the AMR observable expression. default True.
              False is useful for debugging.
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
            - The stacking order requested for the ensemble quantiles to keep the selected
              quantity together for each state.
            - Options: "timepoints" or "quantiles"

    Returns:
        result: dict
            - Dictionary of outputs with following attribute:
                * data: pd.DataFrame
                    - The samples from the model using the optimal policy under uncertainty returned as a DataFrame.
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

    result = prepare_interchange_dictionary(
        samples,
        timepoints,
        interventions=interventions_opt,
        time_unit=time_unit,
        quantiles=True,
        alpha_qs=alpha_qs,
        stacking_order=stacking_order,
        observables=observables,
        visual_options=visual_options,
    )
    result = {**result, "policy": ouu_policy}

    return result


@deprecated(
    "'Big Box' interfaces were a stop-gap for July 2023 hackathon."
    "Please call the appropriate sequence from 'interfaces' insteads."
)
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
    initial_guess: Iterable[float] = [0.5],
    bounds: Iterable[Iterable[float]] = [[0.0], [1.0]],
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
    alpha_qs: Optional[Iterable[float]] = DEFAULT_QUANTILES,
    stacking_order: Optional[str] = "timepoints",
) -> dict:
    """
    Load a petri net from a file, compile it into a probabilistic program,
    calibrate on data, optimize under uncertainty,
    sample for the optimal policy, and estinate risk for the optimal policy.

    Args:
        petri_model_or_path: Union[str, mira.metamodel.TemplateModel, mira.modeling.Model]
            - A path to a petri net file, or a petri net object.
            - This path can be a URL or a local path to a mira model or AMR model.
            - Alternatively, this can be a mira template model directly.
        data_path: str
            - The path to the data to calibrate the model to.
              See notebook/integration_demo/data.csv for an example of the format.
        num_samples: int
            - The number of samples to draw from the model.
        timepoints: [Iterable[float]]
            - The timepoints to simulate the model from.
              Backcasting and/or forecasting is reflected in the choice of timepoints.
        interventions: Iterable[Tuple[float, str]]
            - A list of interventions to apply to the model.
              Each intervention is a tuple of the form (time, parameter_name).
        qoi: Tuple[str, str, **args]
            - Quantity of interest to optimize over.
              QoI is a tuple of the form (callable_qoi_function_name, state_variable_name, function_arguments).
            - Options for "callable_qoi_function_name":
                - scenario2dec_nday_average: performs average over last ndays of timepoints
        risk_bound: float
            - Bound on the risk constraint.
        objfun: callable
            - Objective function as a callable function definition.
              E.g., to minimize the absolute value of intervention parameters use lambda x: np.sum(np.abs(x))
        initial_guess: Iterable[float]
            - Initial guess for the optimizer.
              The length should be equal to number of dimensions of the intervention (or control action).
        bounds: Iterable[Iterable[float]]
            - Lower and upper bounds for intervention parameter.
              Bounds are a list of the form [[lower bounds], [upper bounds]]
        start_state: Optional[dict[str, float]]
            - The initial state of the model. If None, the initial state is taken from the mira model.
        pseudocount: float > 0.0
            - The pseudocount to use for adding uncertainty to the observations.
            - Larger values of pseudocount correspond to more certainty about the observations.
        start_time: float
            - The start time of the model. This is used to align the `start_state` with the `timepoints`.
            - By default we set the `start_time` to be a small negative number to avoid numerical
              issues w/ collision with the `timepoints` which typically start at 0.
        num_iterations: int > 0
            - The number of iterations to run the calibration for.
        lr: float > 0.0
            - The learning rate to use for the calibration.
        verbose: bool
            - Whether to print out the calibration progress and the optimization under uncertainty progress.
              This will include summaries of the evidence lower bound (ELBO) and the parameters.
        num_particles: int > 0
            - The number of particles to use for the calibration.
              Increasing this value will result in lower variance gradient estimates,
            but will also increase the computational cost per gradient step.
        deterministic_learnable_parameters: Iterable[str]
            - The set of parameters whose calibration output will be point estimates that were learned from data.
        method: str
            - The method to use for solving the ODE. See torchdiffeq's `odeint` method for more details.
            - If performance is incredibly slow, we suggest using `euler` to debug.
              If using `euler` results in faster simulation, the issue is likely that the model is stiff.
        compile_rate_law_p: bool
            - Whether or not to compile the rate law from the AMR rate expression. default True.
              False is useful for debugging.
        compile_observables_p: bool
            - Whether or not to compile the observables from the AMR observable expression. default True.
              False is useful for debugging.
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
            - The stacking order requested for the ensemble quantiles to keep
              the selected quantity together for each state.
            - Options: "timepoints" or "quantiles"

    Returns:
        result: dict
            - Dictionary of outputs with following attribute:
                * data: pd.DataFrame
                    - The samples from the model using the optimal policy under uncertainty after
                      calibrating on given data returned as a pandas DataFrame.
                * policy: dict
                    - Optimal policy under uncertainty returned as a dictionary with the following attributes:
                        * policy: Optimal intervention
                        * OptResults: Optimization results as scipy optimization object
                        * risk: Estimated alpha-superquantile risk with alpha=0.95
                        * samples: Samples from the model at the optimal intervention
                        * qoi: Samples of quantity of interest
                * quantiles: The quantiles for ensemble score calculation after calibration as a pandas DataFrames.
                * inferred_parameters: The inferred parameters from the calibration as PetriInferredParameters.
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
        guide.append(
            AutoDelta(
                pyro.poutine.block(model, expose=deterministic_learnable_parameters)
            )
        )
        guide.append(
            AutoLowRankMultivariateNormal(
                pyro.poutine.block(model, hide=deterministic_learnable_parameters)
            )
        )
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
        progress_hook=progress_hook,
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

    result = prepare_interchange_dictionary(
        samples,
        timepoints,
        interventions=interventions_opt,
        time_unit=time_unit,
        quantiles=True,
        alpha_qs=alpha_qs,
        stacking_order=stacking_order,
        observables=observables,
        train_end_point=max([d[0] for d in data]),
        visual_options=visual_options,
    )

    result = {
        **result,
        "policy": ouu_policy,
        "inferred_parameters": inferred_parameters,
    }

    return result
