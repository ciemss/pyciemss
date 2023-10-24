import pyro
import torch
import time
import numpy as np
from math import ceil
from typing import Iterable, Optional, Tuple, Union, Callable
import copy
import warnings

import random as rand

from torch.distributions import biject_to

from pyro.infer import Predictive
from pyro.infer.autoguide import AutoLowRankMultivariateNormal

from pyciemss.PetriNetODE.base import (
    PetriNetODESystem,
    ScaledNormalNoisePetriNetODESystem,
    ScaledBetaNoisePetriNetODESystem,
    get_name,
)
from pyciemss.risk.ouu import computeRisk, solveOUU
from pyciemss.risk.risk_measures import alpha_superquantile
from pyciemss.utils.interface_utils import convert_to_output_format
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
    prepare_interchange_dictionary,
    DEFAULT_QUANTILES
)

from pyciemss.PetriNetODE.events import (
    StartEvent,
    ObservationEvent,
    LoggingEvent,
    StaticParameterInterventionEvent,
)

from pyciemss.custom_decorators import pyciemss_logging_wrapper

# TODO: These interfaces should probably be just in terms of JSON-like objects.

PetriSolution = dict  # NOTE: [str, torch.tensor] type argument removed because of issues with type-based dispatch.
PetriInferredParameters = pyro.nn.PyroModule


def load_petri_model(
    petri_model_or_path: Union[str, mira.metamodel.TemplateModel, mira.modeling.Model],
    add_uncertainty: bool = True,
    noise_model: str = "scaled_normal",
    noise_scale: float = 0.1,
    compile_observables_p: bool = False,
    compile_rate_law_p: bool = False
) -> PetriNetODESystem:
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
    start_state: Optional[dict[str, float]] = None,
) -> PetriNetODESystem:
    """
    Instantiate a model for a particular configuration of initial conditions
    """
    if start_state is None:
        start_state = {
            get_name(v): v.data["initial_value"] for v in petri.G.variables.values()
        }
    
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


@pyciemss_logging_wrapper
def get_posterior_density_mesh_petri(inferred_parameters: PetriInferredParameters,
                                     mesh_params: Optional[dict[str, list[float]]]) -> float:
    """
    Compute the log posterior density of the inferred parameters at the given parameter values.
    Args:
        inferred_parameters: PetriInferredParameters
            - The inferred parameters from the calibration.
        mesh_params: dict[str, list]
            - Parameter values used to compute a mesh of sample points.  
            Keys are parameter names, values are (min, max, steps) parameters passed to linspace.
    Returns:
        log_density: float
            - The log posterior density of the inferred parameters at the given parameter values.
    """    
    spaces = [torch.linspace(*params) for params in mesh_params.values()]
    parameter_values = dict(zip(mesh_params.keys(), torch.meshgrid(*spaces, indexing='ij')))
    density = get_posterior_density_petri(inferred_parameters, parameter_values)
    return parameter_values, density


@pyciemss_logging_wrapper
def get_posterior_density_petri(inferred_parameters: PetriInferredParameters,
                                parameter_values: dict[str, Union[list[float], torch.tensor]]) -> float:
    """
    Compute the log posterior density of the inferred parameters at the given parameter values.
    Args:
        inferred_parameters: PetriInferredParameters
            - The inferred parameters from the calibration.
        parameter_values: dict[str, list]
            - The parameter values to evaluate the log posterior density at.
    Returns:
        log_density: float
            - The log posterior density of the inferred parameters at the given parameter values.
    """

    guides = [guide for guide in inferred_parameters if type(guide) == AutoLowRankMultivariateNormal]

    # By construction there should be only a single AutoLowRankMultivariateNormal guide. The rest should be AutoDeltas.
    if len(guides) != 1:
        raise ValueError(f"Expected a single AutoLowRankMultivariateNormal guide, but found {len(guides)} guides.")

    guide = guides[0]

    # For now we only support density evaluation on the full parameter space.
    if guide.loc.shape[0] != len(parameter_values):
        raise ValueError(f"Expected {guide.loc.shape[0]} parameters, but found {len(parameter_values)} parameters.")

    parameter_values = {name: torch.as_tensor(value) for name, value in parameter_values.items()}

    # Assert that all of the parameters in the `parameter_values` are the same size.
    parameter_sizes = set([value.size() for value in parameter_values.values()])
    if len(parameter_sizes) != 1:
        raise ValueError(f"Expected all parameter values to have the same size, but found {len(parameter_sizes)} distinct sizes.")

    parameter_size = parameter_sizes.pop()

    unconstrained_values = torch.zeros(parameter_size + guide.loc.size())

    for i, (name, site) in enumerate(guide.prototype_trace.iter_stochastic_nodes()):
        transform = biject_to(site["fn"].support)
        value = parameter_values[name]
        unconstrained_value = transform.inv(value)

        unconstrained_values[..., i] = unconstrained_value

    # Compute the log density using the transformed distribution and the unconstrained values.
    log_density = guide.get_posterior().log_prob(unconstrained_values)

    return torch.exp(log_density).detach()

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


@pyciemss_logging_wrapper
@prepare_interchange_dictionary.register
def prepare_interchange_dictionary(
    samples: PetriSolution,
    timepoints: Iterable[float],
    *,
    time_unit: Optional[str] = None,
    alpha_qs: Optional[Iterable[float]] = DEFAULT_QUANTILES,
    stacking_order: Optional[str] = "timepoints",
    visual_options: Union[None, bool, dict[str, any]] = None,
    train_endpoint: Optional[any] = None,
    interventions: Optional[any] = None,
    observables: Optional[any] = None,
) -> dict:
    processed_samples, q_ensemble = convert_to_output_format(
        samples,
        timepoints,
        interventions=interventions,
        time_unit=time_unit,
        quantiles=True,
        alpha_qs=alpha_qs,
        stacking_order=stacking_order,
        observables=observables,
        train_end_point=train_endpoint,
    )

    result = {"data": processed_samples, "quantiles": q_ensemble}

    if visual_options:
        visual_options = {} if visual_options is True else visual_options
        schema = plots.trajectories(processed_samples, **visual_options)
        result["visual"] = schema

    return result
