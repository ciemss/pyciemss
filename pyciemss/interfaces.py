import contextlib
from typing import Any, Callable, Dict, List, Optional, Union

import pyro
import torch
from chirho.dynamical.handlers import (
    DynamicIntervention,
    InterruptionEventLoop,
    LogTrajectory,
    StaticBatchObservation,
    StaticIntervention,
)
from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.ops import State
from chirho.observational.handlers import condition
from pyro.contrib.autoname import scope

from pyciemss.compiled_dynamics import CompiledDynamics
from pyciemss.ensemble.compiled_dynamics import EnsembleCompiledDynamics
from pyciemss.integration_utils.custom_decorators import pyciemss_logging_wrapper
from pyciemss.integration_utils.observation import compile_noise_model
from pyciemss.integration_utils.result_processing import prepare_interchange_dictionary
from pyciemss.ouu.ouu import computeRisk, solveOUU
from pyciemss.ouu.risk_measures import alpha_superquantile

import numpy as np


@pyciemss_logging_wrapper
def ensemble_sample(
    model_paths_or_jsons: List[Union[str, Dict]],
    solution_mappings: List[
        Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
    ],
    end_time: float,
    logging_step_size: float,
    num_samples: int,
    *,
    dirichlet_alpha: Optional[torch.Tensor] = None,
    noise_model: Optional[str] = None,
    noise_model_kwargs: Dict[str, Any] = {},
    solver_method: str = "dopri5",
    solver_options: Dict[str, Any] = {},
    start_time: float = 0.0,
    inferred_parameters: Optional[pyro.nn.PyroModule] = None,
):
    """
    Load a collection of models from files, compile them into an ensemble probabilistic program,
    and sample from the ensemble.

    Args:
    model_paths_or_jsons: List[Union[str, Dict]]
        - A list of paths to AMR model files or JSONs containing models in AMR form.
    solution_mappings: List[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]]
        - A list of functions that map the solution of each model to a common solution space.
        - Each function takes in a dictionary of the form {state_variable_name: value}
            and returns a dictionary of the same form.
    end_time: float
        - The end time of the sampled simulation.
    logging_step_size: float
        - The step size to use for logging the trajectory.
    num_samples: int
        - The number of samples to draw from the model.
    dirichlet_alpha: Optional[torch.Tensor]
        - A tensor of shape (num_models,) containing the Dirichlet alpha values for the ensemble.
        - If not provided, we will use a uniform Dirichlet prior.
    noise_model: Optional[str]
        - The noise model to use for the data.
        - Currently we only support the normal distribution.
    noise_model_kwargs: Dict[str, Any]
        - Keyword arguments to pass to the noise model.
        - Currently we only support the `scale` keyword argument for the normal distribution.
    solver_method: str
        - The method to use for solving the ODE. See torchdiffeq's `odeint` method for more details.
        - If performance is incredibly slow, we suggest using `euler` to debug.
          If using `euler` results in faster simulation, the issue is likely that the model is stiff.
    solver_options: Dict[str, Any]
        - Options to pass to the solver. See torchdiffeq' `odeint` method for more details.
    start_time: float
        - The start time of the model. This is used to align the `start_state` from the
          AMR model with the simulation timepoints.
        - By default we set the `start_time` to be 0.
    inferred_parameters: Optional[pyro.nn.PyroModule]
        - A Pyro module that contains the inferred parameters of the model.
          This is typically the result of `calibrate`.
        - If not provided, we will use the default values from the AMR model.

    Returns:
        result: Dict[str, torch.Tensor]
            - Dictionary of outputs from the model.
                - Each key is the name of a parameter or state variable in the model.
                - Each value is a tensor of shape (num_samples, num_timepoints) for state variables
                    and (num_samples,) for parameters.
    """
    if dirichlet_alpha is None:
        dirichlet_alpha = torch.ones(len(model_paths_or_jsons))

    model = EnsembleCompiledDynamics.load(
        model_paths_or_jsons, dirichlet_alpha, solution_mappings
    )

    timespan = torch.arange(start_time + logging_step_size, end_time, logging_step_size)

    def wrapped_model():
        # We need to interleave the LogTrajectory and the solutions from the models.
        # This because each contituent model will have its own LogTrajectory.

        solutions = [State()] * len(model_paths_or_jsons)

        for i, dynamics in enumerate(model.dynamics_models):
            with scope(prefix=f"model_{i}"):
                with LogTrajectory(timespan) as lt:
                    dynamics(
                        torch.as_tensor(start_time),
                        torch.as_tensor(end_time),
                        TorchDiffEq(method=solver_method, options=solver_options),
                    )

                solutions[i] = lt.trajectory

                # Adding deterministic nodes to the model so that we can access the trajectory in the Predictive object.
                [pyro.deterministic(f"{k}_state", v) for k, v in lt.trajectory.items()]

                if noise_model is not None:
                    compiled_noise_model = compile_noise_model(
                        noise_model,
                        vars=set(lt.trajectory.keys()),
                        **noise_model_kwargs,
                    )
                    # Adding noise to the model so that we can access the noisy trajectory in the Predictive object.
                    compiled_noise_model(lt.trajectory)

        return State(
            **{
                k: sum([model.model_weights[i] * v[k] for i, v in enumerate(solutions)])
                for k in solutions[0].keys()
            }
        )

    samples = pyro.infer.Predictive(
        wrapped_model, guide=inferred_parameters, num_samples=num_samples
    )()

    return prepare_interchange_dictionary(samples)


@pyciemss_logging_wrapper
def sample(
    model_path_or_json: Union[str, Dict],
    end_time: float,
    logging_step_size: float,
    num_samples: int,
    *,
    noise_model: Optional[str] = None,
    noise_model_kwargs: Dict[str, Any] = {},
    solver_method: str = "dopri5",
    solver_options: Dict[str, Any] = {},
    start_time: float = 0.0,
    inferred_parameters: Optional[pyro.nn.PyroModule] = None,
    static_interventions: Dict[float, Dict[str, torch.Tensor]] = {},
    dynamic_interventions: Dict[
        Callable[[Dict[str, torch.Tensor]], torch.Tensor], Dict[str, torch.Tensor]
    ] = {},
) -> Dict[str, torch.Tensor]:
    """
    Load a model from a file, compile it into a probabilistic program, and sample from it.

    Args:
        model_path_or_json: Union[str, Dict]
            - A path to a AMR model file or JSON containing a model in AMR form.
        end_time: float
            - The end time of the sampled simulation.
        logging_step_size: float
            - The step size to use for logging the trajectory.
        num_samples: int
            - The number of samples to draw from the model.
        solver_method: str
            - The method to use for solving the ODE. See torchdiffeq's `odeint` method for more details.
            - If performance is incredibly slow, we suggest using `euler` to debug.
              If using `euler` results in faster simulation, the issue is likely that the model is stiff.
        solver_options: Dict[str, Any]
            - Options to pass to the solver. See torchdiffeq' `odeint` method for more details.
        start_time: float
            - The start time of the model. This is used to align the `start_state` from the
              AMR model with the simulation timepoints.
            - By default we set the `start_time` to be 0.
        inferred_parameters: Optional[pyro.nn.PyroModule]
            - A Pyro module that contains the inferred parameters of the model.
              This is typically the result of `calibrate`.
            - If not provided, we will use the default values from the AMR model.
        static_interventions: Dict[float, Dict[str, torch.Tensor]]
            - A dictionary of static interventions to apply to the model.
            - Each key is the time at which the intervention is applied.
            - Each value is a dictionary of the form {state_variable_name: value}.
        dynamic_interventions: Dict[Callable[[Dict[str, torch.Tensor]], torch.Tensor], Dict[str, torch.Tensor]]
            - A dictionary of dynamic interventions to apply to the model.
            - Each key is a function that takes in the current state of the model and returns a tensor.
              When this function crosses 0, the dynamic intervention is applied.
            - Each value is a dictionary of the form {state_variable_name: value}.

    Returns:
        result: Dict[str, torch.Tensor]
            - Dictionary of outputs from the model.
                - Each key is the name of a parameter or state variable in the model.
                - Each value is a tensor of shape (num_samples, num_timepoints) for state variables
                    and (num_samples,) for parameters.
    """

    model = CompiledDynamics.load(model_path_or_json)

    timespan = torch.arange(start_time + logging_step_size, end_time, logging_step_size)

    static_intervention_handlers = [
        StaticIntervention(time, State(**static_intervention_assignment))
        for time, static_intervention_assignment in static_interventions.items()
    ]
    dynamic_intervention_handlers = [
        DynamicIntervention(event_fn, State(**dynamic_intervention_assignment))
        for event_fn, dynamic_intervention_assignment in dynamic_interventions.items()
    ]

    def wrapped_model():
        with LogTrajectory(timespan) as lt:
            with InterruptionEventLoop():
                with contextlib.ExitStack() as stack:
                    for handler in (
                        static_intervention_handlers + dynamic_intervention_handlers
                    ):
                        stack.enter_context(handler)
                    model(
                        torch.as_tensor(start_time),
                        torch.as_tensor(end_time),
                        TorchDiffEq(method=solver_method, options=solver_options),
                    )

        trajectory = model.add_observables(lt.trajectory)

        # Adding deterministic nodes to the model so that we can access the trajectory in the Predictive object.
        [pyro.deterministic(k, v) for k, v in trajectory.items()]

        if noise_model is not None:
            compiled_noise_model = compile_noise_model(
                noise_model, vars=set(trajectory.keys()), **noise_model_kwargs
            )
            # Adding noise to the model so that we can access the noisy trajectory in the Predictive object.
            compiled_noise_model(trajectory)

    samples = pyro.infer.Predictive(
        wrapped_model, guide=inferred_parameters, num_samples=num_samples
    )()

    return prepare_interchange_dictionary(samples)


def calibrate(
    model_path_or_json: Union[str, Dict],
    data: Dict[str, torch.Tensor],
    data_timepoints: torch.Tensor,
    *,
    noise_model: str = "normal",
    noise_model_kwargs: Dict[str, Any] = {"scale": 0.1},
    solver_method: str = "dopri5",
    solver_options: Dict[str, Any] = {},
    start_time: float = 0.0,
    static_interventions: Dict[float, Dict[str, torch.Tensor]] = {},
    dynamic_interventions: Dict[
        Callable[[State[torch.Tensor]], torch.Tensor], Dict[str, torch.Tensor]
    ] = {},
    num_iterations: int = 1000,
    lr: float = 0.03,
    verbose: bool = False,
    num_particles: int = 1,
    deterministic_learnable_parameters: List[str] = [],
) -> pyro.nn.PyroModule:
    """
    Infer parameters for a DynamicalSystem model conditional on data.
    This uses variational inference with a mean-field variational family to infer the parameters of the model.

    Args:
        - model_path_or_json: Union[str, Dict]
            - A path to a AMR model file or JSON containing a model in AMR form.
        - data: Dict[str, torch.Tensor]
            - A dictionary of data to condition the model on.
            - Each key is the name of a state variable in the model.
            - Each value is a tensor of shape (num_timepoints,) for state variables.
        - data_timepoints: torch.Tensor
            - A tensor of shape (num_timepoints,) containing the timepoints for the data.
        - noise_model: str
            - The noise model to use for the data.
            - Currently we only support the normal distribution.
        - noise_model_kwargs: Dict[str, Any]
            - Keyword arguments to pass to the noise model.
            - Currently we only support the `scale` keyword argument for the normal distribution.
        - solver_method: str
            - The method to use for solving the ODE. See torchdiffeq's `odeint` method for more details.
            - If performance is incredibly slow, we suggest using `euler` to debug.
              If using `euler` results in faster simulation, the issue is likely that the model is stiff.
        - solver_options: Dict[str, Any]
            - Options to pass to the solver. See torchdiffeq' `odeint` method for more details.
        - start_time: float
            - The start time of the model. This is used to align the `start_state` from the
              AMR model with the simulation timepoints.
            - By default we set the `start_time` to be 0.
        - static_interventions: Dict[float, Dict[str, torch.Tensor]]
            - A dictionary of static interventions to apply to the model.
            - Each key is the time at which the intervention is applied.
            - Each value is a dictionary of the form {state_variable_name: value}.
        - dynamic_interventions: Dict[Callable[[Dict[str, torch.Tensor]], torch.Tensor], Dict[str, torch.Tensor]]
            - A dictionary of dynamic interventions to apply to the model.
            - Each key is a function that takes in the current state of the model and returns a tensor.
              When this function crosses 0, the dynamic intervention is applied.
            - Each value is a dictionary of the form {state_variable_name: value}.
        - num_iterations: int
            - The number of iterations to run the inference algorithm for.
        - lr: float
            - The learning rate to use for the inference algorithm.
        - verbose: bool
            - Whether to print out the loss at each iteration.
        - num_particles: int
            - The number of particles to use for the inference algorithm.
        - deterministic_learnable_parameters: List[str]
            - A list of parameter names that should be learned deterministically.
            - By default, all parameters are learned probabilistically.

    Returns:
        - inferred_parameters: pyro.nn.PyroModule
            - A Pyro module that contains the inferred parameters of the model.
            - This can be passed to `sample` to sample from the model conditional on the data.
    """

    model = CompiledDynamics.load(model_path_or_json)

    def autoguide(model):
        guide = pyro.infer.autoguide.AutoGuideList(model)
        guide.append(
            pyro.infer.autoguide.AutoDelta(
                pyro.poutine.block(model, expose=deterministic_learnable_parameters)
            )
        )

        try:
            mvn_guide = pyro.infer.autoguide.AutoLowRankMultivariateNormal(
                pyro.poutine.block(model, hide=deterministic_learnable_parameters)
            )
            mvn_guide._setup_prototype()
            guide.append(mvn_guide)
        except RuntimeError as re:
            assert (
                re.args[0]
                == "AutoLowRankMultivariateNormal found no latent variables; Use an empty guide instead"
            )

        return guide

    static_intervention_handlers = [
        StaticIntervention(time, State(**static_intervention_assignment))
        for time, static_intervention_assignment in static_interventions.items()
    ]
    dynamic_intervention_handlers = [
        DynamicIntervention(event_fn, State(**dynamic_intervention_assignment))
        for event_fn, dynamic_intervention_assignment in dynamic_interventions.items()
    ]

    _noise_model = compile_noise_model(
        noise_model, vars=set(data.keys()), **noise_model_kwargs
    )

    _data = {f"{k}_observed": v for k, v in data.items()}

    def wrapped_model():
        # TODO: pick up here.
        obs = condition(data=_data)(_noise_model)

        with StaticBatchObservation(data_timepoints, observation=obs):
            with InterruptionEventLoop():
                with contextlib.ExitStack() as stack:
                    for handler in (
                        static_intervention_handlers + dynamic_intervention_handlers
                    ):
                        stack.enter_context(handler)
                    model(
                        torch.as_tensor(start_time),
                        torch.as_tensor(data_timepoints[-1]),
                        TorchDiffEq(method=solver_method, options=solver_options),
                    )

    inferred_parameters = autoguide(wrapped_model)

    optim = pyro.optim.Adam({"lr": lr})
    loss = pyro.infer.Trace_ELBO(num_particles=num_particles)
    svi = pyro.infer.SVI(wrapped_model, inferred_parameters, optim, loss=loss)

    pyro.clear_param_store()

    for i in range(num_iterations):
        loss = svi.step()
        if verbose:
            if i % 25 == 0:
                print(f"iteration {i}: loss = {loss}")

    return inferred_parameters


# # TODO
# def optimize(
#     model: CompiledDynamics,
#     objective_function: ObjectiveFunction,
#     constraints: Constraints,
#     optimization_algorithm: OptimizationAlgorithm,
#     *args,
#     **kwargs
# ) -> OptimizationResult:
#     """
#     Optimize the objective function subject to the constraints.
#     """
#     raise NotImplementedError
def optimize(
    model_path_or_json: Union[str, Dict],
    end_time: float,
    qoi: callable,
    risk_bound: float,
    objfun: callable = lambda x: np.abs(x),
    initial_guess: List[float] = 0.5,
    bounds: List[List[float]] = [[0.0], [1.0]],
    *,
    noise_model: str = "normal",
    noise_model_kwargs: Dict[str, Any] = {"scale": 0.1},
    solver_method: str = "dopri5",
    solver_options: Dict[str, Any] = {},
    start_time: float = 0.0,
    inferred_parameters: Optional[pyro.nn.PyroModule] = None,
    n_samples_ouu: int = int(1e2),
    maxiter: int = 2,
    maxfeval: int = 25,
    static_interventions: Dict[float, Dict[str, torch.Tensor]] = {},
    dynamic_interventions: Dict[
        Callable[[State[torch.Tensor]], torch.Tensor], Dict[str, torch.Tensor]
    ] = {},
    verbose: bool = False,
    # petri: PetriNetODESystem,
    # timepoints: Iterable,
    # interventions: dict,
    # method="dopri5",
    roundup_decimal: int = 4,
    postprocess: bool = False,
) -> Dict:
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