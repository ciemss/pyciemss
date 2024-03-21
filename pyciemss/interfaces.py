import contextlib
import time
import warnings
from math import ceil
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pyro
import torch
from chirho.dynamical.handlers import (
    DynamicIntervention,
    StaticBatchObservation,
    StaticIntervention,
)
from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.interventional.ops import Intervention
from chirho.observational.handlers import condition
from chirho.observational.ops import observe

from pyciemss.compiled_dynamics import CompiledDynamics
from pyciemss.ensemble.compiled_dynamics import EnsembleCompiledDynamics
from pyciemss.integration_utils.custom_decorators import pyciemss_logging_wrapper
from pyciemss.integration_utils.interface_checks import check_solver
from pyciemss.integration_utils.observation import compile_noise_model, load_data
from pyciemss.integration_utils.result_processing import prepare_interchange_dictionary
from pyciemss.interruptions import (
    DynamicParameterIntervention,
    ParameterInterventionTracer,
    StaticParameterIntervention,
)
from pyciemss.ouu.ouu import computeRisk, solveOUU
from pyciemss.ouu.risk_measures import alpha_superquantile

warnings.simplefilter("always", UserWarning)


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
    time_unit: Optional[str] = None,
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
            - A higher proportion of alpha values will result in higher weights for the corresponding models.
            - A larger total alpha values will result in more certain priors.
            - e.g. torch.tensor([1, 1, 1]) will result in a uniform prior over vectors of length 3 that sum to 1.
            - e.g. torch.tensor([1, 2, 3]) will result in a prior that is biased towards the third model.
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
    check_solver(solver_method, solver_options)

    with torch.no_grad():
        if dirichlet_alpha is None:
            dirichlet_alpha = torch.ones(len(model_paths_or_jsons))

        model = EnsembleCompiledDynamics.load(
            model_paths_or_jsons, dirichlet_alpha, solution_mappings
        )

        logging_times = torch.arange(
            start_time + logging_step_size, end_time, logging_step_size
        )

        # Check that num_samples is a positive integer
        if not (isinstance(num_samples, int) and num_samples > 0):
            raise ValueError("num_samples must be a positive integer")

        def wrapped_model():
            with TorchDiffEq(method=solver_method, options=solver_options):
                solution = model(
                    torch.as_tensor(start_time),
                    torch.as_tensor(end_time),
                    logging_times=logging_times,
                    is_traced=True,
                )

            if noise_model is not None:
                compiled_noise_model = compile_noise_model(
                    noise_model,
                    vars=set(solution.keys()),
                    **noise_model_kwargs,
                )
                # Adding noise to the model so that we can access the noisy trajectory in the trace.
                compiled_noise_model(solution)

            return solution

        samples = pyro.infer.Predictive(
            wrapped_model,
            guide=inferred_parameters,
            num_samples=num_samples,
            parallel=True,
        )()

        return prepare_interchange_dictionary(
            samples, timepoints=logging_times, time_unit=time_unit
        )


@pyciemss_logging_wrapper
def ensemble_calibrate(
    model_paths_or_jsons: List[Union[str, Dict]],
    solution_mappings: List[
        Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
    ],
    data_path: str,
    *,
    dirichlet_alpha: Optional[torch.Tensor] = None,
    data_mapping: Dict[str, str] = {},
    noise_model: str = "normal",
    noise_model_kwargs: Dict[str, Any] = {"scale": 0.1},
    solver_method: str = "dopri5",
    solver_options: Dict[str, Any] = {},
    start_time: float = 0.0,
    num_iterations: int = 1000,
    lr: float = 0.03,
    verbose: bool = False,
    num_particles: int = 1,
    deterministic_learnable_parameters: List[str] = [],
    progress_hook: Callable = lambda i, loss: None,
) -> Dict[str, Any]:
    """
    Infer parameters for an ensemble of DynamicalSystem models conditional on data.
    This uses variational inference with a mean-field variational family to infer the parameters of the model.

    Args:
    model_paths_or_jsons: List[Union[str, Dict]]
        - A list of paths to AMR model files or JSONs containing models in AMR form.
    solution_mappings: List[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]]
        - A list of functions that map the solution of each model to a common solution space.
        - Each function takes in a dictionary of the form {state_variable_name: value}
            and returns a dictionary of the same form.
    data_path: str
        - A path to the data file.
    dirichlet_alpha: Optional[torch.Tensor]
        - A tensor of shape (num_models,) containing the Dirichlet alpha values for the ensemble.
            - A higher proportion of alpha values will result in higher weights for the corresponding models.
            - A larger total alpha values will result in more certain priors.
            - e.g. torch.tensor([1, 1, 1]) will result in a uniform prior over vectors of length 3 that sum to 1.
            - e.g. torch.tensor([1, 2, 3]) will result in a prior that is biased towards the third model.
        - If not provided, we will use a uniform Dirichlet prior.
    data_mapping: Dict[str, str]
        - A mapping from column names in the data file to state variable names in the model.
            - keys: str name of column in dataset
            - values: str name of state/observable in model
        - If not provided, we will assume that the column names in the data file match the state variable names.
        - Note: This mapping must match output of `solution_mappings`.
    noise_model: str
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
    num_iterations: int
        - The number of iterations to run the inference algorithm for.
    lr: float
        - The learning rate to use for the inference algorithm.
    verbose: bool
        - Whether to print out the loss at each iteration.
    num_particles: int
        - The number of particles to use for the inference algorithm.
    deterministic_learnable_parameters: List[str]
        - A list of parameter names that should be learned deterministically.
        - By default, all parameters are learned probabilistically.
    progress_hook: Callable[[int, float], None]
        - A function that takes in the current iteration and the current loss.
        - This is called at the beginning of each iteration.
        - By default, this is a no-op.
        - This can be used to implement custom progress bars.

    Returns:
        result: Dict[str, Any]
            - Dictionary with the following key-value pairs.
                - inferred_parameters: pyro.nn.PyroModule
                    - A Pyro module that contains the inferred parameters of the model.
                    - This can be passed to `ensemble_sample` to sample from the model conditional on the data.
                - loss: float
                    - The final loss value of the approximate ELBO loss.
    """

    pyro.clear_param_store()

    if dirichlet_alpha is None:
        dirichlet_alpha = torch.ones(len(model_paths_or_jsons))

    model = EnsembleCompiledDynamics.load(
        model_paths_or_jsons, dirichlet_alpha, solution_mappings
    )

    data_timepoints, data = load_data(data_path, data_mapping=data_mapping)

    # Check that num_iterations is a positive integer
    if not (isinstance(num_iterations, int) and num_iterations > 0):
        raise ValueError("num_iterations must be a positive integer")

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
            if (
                re.args[0]
                != "AutoLowRankMultivariateNormal found no latent variables; Use an empty guide instead"
            ):
                raise re

        return guide

    _noise_model = compile_noise_model(
        noise_model,
        vars=set(data.keys()),
        **noise_model_kwargs,
    )

    _data = {f"{k}_noisy": v for k, v in data.items()}

    def wrapped_model():
        obs = condition(data=_data)(_noise_model)

        with TorchDiffEq(method=solver_method, options=solver_options):
            solution = model(
                torch.as_tensor(start_time),
                torch.as_tensor(data_timepoints[-1]),
                logging_times=data_timepoints,
                is_traced=True,
            )

            observe(solution, obs)

    inferred_parameters = autoguide(wrapped_model)

    optim = pyro.optim.Adam({"lr": lr})
    loss = pyro.infer.Trace_ELBO(num_particles=num_particles)
    svi = pyro.infer.SVI(wrapped_model, inferred_parameters, optim, loss=loss)

    for i in range(num_iterations):
        # Call a progress hook at the beginning of each iteration. This is used to implement custom progress bars.
        progress_hook(i, loss)
        loss = svi.step()
        if verbose:
            if i % 25 == 0:
                print(f"iteration {i}: loss = {loss}")

    return {"inferred_parameters": inferred_parameters, "loss": loss}


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
    time_unit: Optional[str] = None,
    inferred_parameters: Optional[pyro.nn.PyroModule] = None,
    static_state_interventions: Dict[torch.Tensor, Dict[str, Intervention]] = {},
    static_parameter_interventions: Dict[torch.Tensor, Dict[str, Intervention]] = {},
    dynamic_state_interventions: Dict[
        Callable[[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor],
        Dict[str, Intervention],
    ] = {},
    dynamic_parameter_interventions: Dict[
        Callable[[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor],
        Dict[str, Intervention],
    ] = {},
    alpha: float = 0.95,
) -> Dict[str, Any]:
    r"""
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
        static_state_interventions: Dict[float, Dict[str, Intervention]]
            - A dictionary of static interventions to apply to the model.
            - Each key is the time at which the intervention is applied.
            - Each value is a dictionary of the form {state_variable_name: intervention_assignment}.
            - Note that the `intervention_assignment` can be any type supported by
              :func:`~chirho.interventional.ops.intervene`, including functions.
        static_parameter_interventions: Dict[float, Dict[str, Intervention]]
            - A dictionary of static interventions to apply to the model.
            - Each key is the time at which the intervention is applied.
            - Each value is a dictionary of the form {parameter_name: intervention_assignment}.
            - Note that the `intervention_assignment` can be any type supported by
              :func:`~chirho.interventional.ops.intervene`, including functions.
        dynamic_state_interventions: Dict[
                                        Callable[[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor],
                                        Dict[str, Intervention]
                                        ]
            - A dictionary of dynamic interventions to apply to the model.
            - Each key is a function that takes in the current state of the model and returns a tensor.
              When this function crosses 0, the dynamic intervention is applied.
            - Each value is a dictionary of the form {state_variable_name: intervention_assignment}.
            - Note that the `intervention_assignment` can be any type supported by
              :func:`~chirho.interventional.ops.intervene`, including functions.
        dynamic_parameter_interventions: Dict[
                                            Callable[[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor],
                                            Dict[str, Intervention]
                                            ]
            - A dictionary of dynamic interventions to apply to the model.
            - Each key is a function that takes in the current state of the model and returns a tensor.
              When this function crosses 0, the dynamic intervention is applied.
            - Each value is a dictionary of the form {parameter_name: intervention_assignment}.
            - Note that the `intervention_assignment` can be any type supported by
              :func:`~chirho.interventional.ops.intervene`, including functions.
        alpha: float
            - Risk level for alpha-superquantile outputs in the results dictionary.

    Returns:
        result: Dict[str, torch.Tensor]
            - Dictionary of outputs with following attributes:
                - data: The samples from the model as a pandas DataFrame.
                - unprocessed_result: Dictionary of outputs from the model.
                    - Each key is the name of a parameter or state variable in the model.
                    - Each value is a tensor of shape (num_samples, num_timepoints) for state variables
                    and (num_samples,) for parameters.
                - quantiles: The quantiles for ensemble score calculation as a pandas DataFrames.
                - risk: Dictionary with each key as the name of a state with
                a dictionary of risk estimates for each state at the final timepoint.
                    - risk: alpha-superquantile risk estimate
                    Superquantiles can be intuitively thought of as a tail expectation, or an average
                    over a portion of worst-case outcomes. Given a distribution of a
                    quantity of interest (QoI), the superquantile at level \alpha\in[0, 1] is
                    the expected value of the largest 100(1 -\alpha)% realizations of the QoI.
                    - qoi: Samples of quantity of interest (value of the state at the final timepoint)
                - schema: Visualization. (If visual_options is truthy)
    """

    check_solver(solver_method, solver_options)

    with torch.no_grad():
        model = CompiledDynamics.load(model_path_or_json)

        logging_times = torch.arange(
            start_time + logging_step_size, end_time, logging_step_size
        )

        # Check that num_samples is a positive integer
        if not (isinstance(num_samples, int) and num_samples > 0):
            raise ValueError("num_samples must be a positive integer")

        static_state_intervention_handlers = [
            StaticIntervention(time, dict(**static_intervention_assignment))
            for time, static_intervention_assignment in static_state_interventions.items()
        ]
        static_parameter_intervention_handlers = [
            StaticParameterIntervention(
                time, dict(**static_intervention_assignment), is_traced=True
            )
            for time, static_intervention_assignment in static_parameter_interventions.items()
        ]

        dynamic_state_intervention_handlers = [
            DynamicIntervention(event_fn, dict(**dynamic_intervention_assignment))
            for event_fn, dynamic_intervention_assignment in dynamic_state_interventions.items()
        ]

        dynamic_parameter_intervention_handlers = [
            DynamicParameterIntervention(
                event_fn, dict(**dynamic_intervention_assignment), is_traced=True
            )
            for event_fn, dynamic_intervention_assignment in dynamic_parameter_interventions.items()
        ]

        intervention_handlers = (
            static_state_intervention_handlers
            + static_parameter_intervention_handlers
            + dynamic_state_intervention_handlers
            + dynamic_parameter_intervention_handlers
        )

        def wrapped_model():
            with ParameterInterventionTracer():
                with TorchDiffEq(method=solver_method, options=solver_options):
                    with contextlib.ExitStack() as stack:
                        for handler in intervention_handlers:
                            stack.enter_context(handler)
                        full_trajectory = model(
                            torch.as_tensor(start_time),
                            torch.as_tensor(end_time),
                            logging_times=logging_times,
                            is_traced=True,
                        )

            if noise_model is not None:
                compiled_noise_model = compile_noise_model(
                    noise_model,
                    vars=set(full_trajectory.keys()),
                    observables=model.observables,
                    **noise_model_kwargs,
                )
                # Adding noise to the model so that we can access the noisy trajectory in the Predictive object.
                compiled_noise_model(full_trajectory)

        parallel = (
            False
            if len(
                dynamic_parameter_intervention_handlers
                + dynamic_state_intervention_handlers
            )
            > 0
            else True
        )

        samples = pyro.infer.Predictive(
            wrapped_model,
            guide=inferred_parameters,
            num_samples=num_samples,
            parallel=parallel,
        )()

        risk_results = {}
        for k, vals in samples.items():
            if "_state" in k:
                # qoi is assumed to be the last day of simulation
                qoi_sample = vals.detach().numpy()[:, -1]
                sq_est = alpha_superquantile(qoi_sample, alpha=alpha)
                risk_results.update({k: {"risk": [sq_est], "qoi": qoi_sample}})

        return {
            **prepare_interchange_dictionary(
                samples, timepoints=logging_times, time_unit=time_unit
            ),
            "risk": risk_results,
        }


@pyciemss_logging_wrapper
def calibrate(
    model_path_or_json: Union[str, Dict],
    data_path: str,
    *,
    data_mapping: Dict[str, str] = {},
    noise_model: str = "normal",
    noise_model_kwargs: Dict[str, Any] = {"scale": 0.1},
    solver_method: str = "dopri5",
    solver_options: Dict[str, Any] = {},
    start_time: float = 0.0,
    static_state_interventions: Dict[torch.Tensor, Dict[str, Intervention]] = {},
    static_parameter_interventions: Dict[torch.Tensor, Dict[str, Intervention]] = {},
    dynamic_state_interventions: Dict[
        Callable[[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor],
        Dict[str, Intervention],
    ] = {},
    dynamic_parameter_interventions: Dict[
        Callable[[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor],
        Dict[str, Intervention],
    ] = {},
    num_iterations: int = 1000,
    lr: float = 0.03,
    verbose: bool = False,
    num_particles: int = 1,
    deterministic_learnable_parameters: List[str] = [],
    progress_hook: Callable = lambda i, loss: None,
) -> Dict[str, Any]:
    """
    Infer parameters for a DynamicalSystem model conditional on data.
    This uses variational inference with a mean-field variational family to infer the parameters of the model.

    Args:
        - model_path_or_json: Union[str, Dict]
            - A path to a AMR model file or JSON containing a model in AMR form.
        - data_path: str
            - A path to the data file.
        - data_mapping: Dict[str, str]
            - A mapping from column names in the data file to state variable names in the model.
                - keys: str name of column in dataset
                - values: str name of state/observable in model
            - If not provided, we will assume that the column names in the data file match the state variable names.
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
        static_state_interventions: Dict[float, Dict[str, Intervention]]
            - A dictionary of static interventions to apply to the model.
            - Each key is the time at which the intervention is applied.
            - Each value is a dictionary of the form {state_variable_name: intervention_assignment}.
            - Note that the `intervention_assignment` can be any type supported by
              :func:`~chirho.interventional.ops.intervene`, including functions.
        static_parameter_interventions: Dict[float, Dict[str, Intervention]]
            - A dictionary of static interventions to apply to the model.
            - Each key is the time at which the intervention is applied.
            - Each value is a dictionary of the form {parameter_name: intervention_assignment}.
            - Note that the `intervention_assignment` can be any type supported by
              :func:`~chirho.interventional.ops.intervene`, including functions.
        dynamic_state_interventions: Dict[
                                        Callable[[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor],
                                        Dict[str, Intervention]
                                        ]
            - A dictionary of dynamic interventions to apply to the model.
            - Each key is a function that takes in the current state of the model and returns a tensor.
              When this function crosses 0, the dynamic intervention is applied.
            - Each value is a dictionary of the form {state_variable_name: intervention_assignment}.
            - Note that the `intervention_assignment` can be any type supported by
              :func:`~chirho.interventional.ops.intervene`, including functions.
        dynamic_parameter_interventions: Dict[
                                            Callable[[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor],
                                            Dict[str, Intervention]
                                            ]
            - A dictionary of dynamic interventions to apply to the model.
            - Each key is a function that takes in the current state of the model and returns a tensor.
              When this function crosses 0, the dynamic intervention is applied.
            - Each value is a dictionary of the form {parameter_name: intervention_assignment}.
            - Note that the `intervention_assignment` can be any type supported by
              :func:`~chirho.interventional.ops.intervene`, including functions.
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
        - progress_hook: Callable[[int, float], None]
            - A function that takes in the current iteration and the current loss.
            - This is called at the beginning of each iteration.
            - By default, this is a no-op.
            - This can be used to implement custom progress bars.

    Returns:
        result: Dict[str, Any]
            - Dictionary with the following key-value pairs.
                - inferred_parameters: pyro.nn.PyroModule
                    - A Pyro module that contains the inferred parameters of the model.
                    - This can be passed to `sample` to sample from the model conditional on the data.
                - loss: float
                    - The final loss value of the approximate ELBO loss.
    """

    check_solver(solver_method, solver_options)

    pyro.clear_param_store()

    model = CompiledDynamics.load(model_path_or_json)

    data_timepoints, data = load_data(data_path, data_mapping=data_mapping)

    # Check that num_iterations is a positive integer
    if not (isinstance(num_iterations, int) and num_iterations > 0):
        raise ValueError("num_iterations must be a positive integer")

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
            if (
                re.args[0]
                != "AutoLowRankMultivariateNormal found no latent variables; Use an empty guide instead"
            ):
                raise re

        return guide

    static_state_intervention_handlers = [
        StaticIntervention(time, dict(**static_intervention_assignment))
        for time, static_intervention_assignment in static_state_interventions.items()
    ]
    static_parameter_intervention_handlers = [
        StaticParameterIntervention(time, dict(**static_intervention_assignment))
        for time, static_intervention_assignment in static_parameter_interventions.items()
    ]

    dynamic_state_intervention_handlers = [
        DynamicIntervention(event_fn, dict(**dynamic_intervention_assignment))
        for event_fn, dynamic_intervention_assignment in dynamic_state_interventions.items()
    ]

    dynamic_parameter_intervention_handlers = [
        DynamicParameterIntervention(event_fn, dict(**dynamic_intervention_assignment))
        for event_fn, dynamic_intervention_assignment in dynamic_parameter_interventions.items()
    ]

    intervention_handlers = (
        static_state_intervention_handlers
        + static_parameter_intervention_handlers
        + dynamic_state_intervention_handlers
        + dynamic_parameter_intervention_handlers
    )

    _noise_model = compile_noise_model(
        noise_model,
        vars=set(data.keys()),
        observables=model.observables,
        **noise_model_kwargs,
    )

    _data = {f"{k}_noisy": v for k, v in data.items()}

    def wrapped_model():
        obs = condition(data=_data)(_noise_model)

        with StaticBatchObservation(data_timepoints, observation=obs):
            with TorchDiffEq(method=solver_method, options=solver_options):
                with contextlib.ExitStack() as stack:
                    for handler in intervention_handlers:
                        stack.enter_context(handler)
                    model(
                        torch.as_tensor(start_time),
                        torch.as_tensor(data_timepoints[-1]),
                    )

    inferred_parameters = autoguide(wrapped_model)

    optim = pyro.optim.Adam({"lr": lr})
    loss = pyro.infer.Trace_ELBO(num_particles=num_particles)
    svi = pyro.infer.SVI(wrapped_model, inferred_parameters, optim, loss=loss)

    pyro.clear_param_store()

    for i in range(num_iterations):
        # Call a progress hook at the beginning of each iteration. This is used to implement custom progress bars.
        progress_hook(i, loss)
        loss = svi.step()
        if verbose:
            if i % 25 == 0:
                print(f"iteration {i}: loss = {loss}")

    return {"inferred_parameters": inferred_parameters, "loss": loss}


def optimize(
    model_path_or_json: Union[str, Dict],
    end_time: float,
    logging_step_size: float,
    qoi: Callable,
    risk_bound: float,
    static_parameter_interventions: Callable[
        [torch.Tensor], Dict[float, Dict[str, Intervention]]
    ],
    objfun: Callable,
    initial_guess_interventions: List[float],
    bounds_interventions: List[List[float]],
    *,
    alpha: float = 0.95,
    solver_method: str = "dopri5",
    solver_options: Dict[str, Any] = {},
    start_time: float = 0.0,
    inferred_parameters: Optional[pyro.nn.PyroModule] = None,
    n_samples_ouu: int = int(1e3),
    maxiter: int = 5,
    maxfeval: int = 25,
    verbose: bool = False,
    roundup_decimal: int = 4,
) -> Dict[str, Any]:
    r"""
    Load a model from a file, compile it into a probabilistic program, and optimize under uncertainty with risk-based
    constraints over dynamical models. This uses \alpha-superquantile as the risk measure. Superquantiles can be
    intuitively thought of as a tail expectation, or an average over a portion of worst-case outcomes. Given a
    distribution of a quantity of interest (QoI), the superquantile at level \alpha\in[0, 1] is the expected
    value of the largest 100(1 -\alpha)% realizations of the QoI.
    Args:
        model_path_or_json: Union[str, Dict]
            - A path to a AMR model file or JSON containing a model in AMR form.
        end_time: float
            - The end time of the sampled simulation.
        logging_step_size: float
            - The step size to use for logging the trajectory.
        qoi: Callable
            - A callable function defining the quantity of interest to optimize over.
        risk_bounds: float
            - The threshold on the risk constraint.
        static_parameter_interventions: Callable[[torch.Tensor], Dict[float, Dict[str, Intervention]]]
            - A callable function of static parameter interventions to optimize over.
            - The callable functions are created using the provided templates:
                - param_value_objective(): creates a static parameter intervention when optimizing over
                (multiple) parameter values
                - start_time_objective(): creates a static parameter intervention when optimizing over
                (multiple) start times for different parameter
        objfun: Callable
            - The objective function defined as a callable function definition.
            - E.g., to minimize the absolute value of intervention parameters use lambda x: np.sum(np.abs(x))
        initial_guess_interventions: List[float]
            - The initial guess for the optimizer.
            - The length should be equal to number of dimensions of the intervention (or control action).
        bounds_interventions: List[List[float]]
            - The lower and upper bounds for intervention parameter.
            - Bounds are a list of the form [[lower bounds], [upper bounds]]
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
        n_samples_ouu: int
            - The number of samples to draw from the model to estimate risk for each optimization iteration.
        maxiter: int
            - Maximum number of basinhopping iterations: >0 leads to multi-start
        maxfeval: int
            - Maximum number of function evaluations for each local optimization step
        verbose: bool
            - Whether to print out the optimization under uncertainty progress.
        roundup_decimal: int
            - Number of significant digits for the optimal policy.

    Returns:
        result: Dict[str, Any]
            - Dictionary with the following key-value pairs.
                - policy: torch.tensor(opt_results.x)
                    - Optimal intervention as the solution of the optimization under uncertainty problem.
                - OptResults: scipy OptimizeResult object
                    - Optimization results as scipy object.
    """
    check_solver(solver_method, solver_options)

    with torch.no_grad():
        control_model = CompiledDynamics.load(model_path_or_json)
        bounds_np = np.atleast_2d(bounds_interventions)
        u_min = bounds_np[0, :]
        u_max = bounds_np[1, :]
        # Set up risk estimation
        RISK = computeRisk(
            model=control_model,
            interventions=static_parameter_interventions,
            qoi=qoi,
            end_time=end_time,
            logging_step_size=logging_step_size,
            start_time=start_time,
            risk_measure=lambda z: alpha_superquantile(z, alpha=alpha),
            num_samples=1,
            guide=inferred_parameters,
            solver_method=solver_method,
            solver_options=solver_options,
            u_bounds=bounds_np,
            risk_bound=risk_bound,
        )

        # Run one sample to estimate model evaluation time
        start_t = time.time()
        init_prediction = RISK.propagate_uncertainty(initial_guess_interventions)
        RISK.qoi(init_prediction)
        end_t = time.time()
        forward_time = end_t - start_t
        time_per_eval = forward_time / 1.0
        if verbose:
            print(f"Time taken: ({forward_time/1.:.2e} seconds per model evaluation).")

        # Assign the required number of MC samples for each OUU iteration
        RISK = computeRisk(
            model=control_model,
            interventions=static_parameter_interventions,
            qoi=qoi,
            end_time=end_time,
            logging_step_size=logging_step_size,
            start_time=start_time,
            risk_measure=lambda z: alpha_superquantile(z, alpha=alpha),
            num_samples=n_samples_ouu,
            guide=inferred_parameters,
            solver_method=solver_method,
            solver_options=solver_options,
            u_bounds=bounds_np,
            risk_bound=risk_bound,
        )
        # Define constraints >= 0
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

        # Updating the objective function to penalize out of bounds interventions
        def objfun_penalty(x):
            if np.any(x - u_min < 0) or np.any(u_max - x < 0):
                return objfun(x) + max(5 * np.abs(objfun(x)), 5.0)
            else:
                return objfun(x)

        start_time = time.time()
        opt_results = solveOUU(
            x0=initial_guess_interventions,
            objfun=objfun_penalty,
            constraints=constraints,
            maxiter=maxiter,
            maxfeval=maxfeval,
            u_bounds=bounds_np,
        ).solve()

        # Rounding up to given number of decimal places
        def round_up(num, dec=roundup_decimal):
            rnum = np.zeros(num.shape[-1])
            for i in range(num.shape[-1]):
                rnum[i] = ceil(num[i] * 10**dec) / (10**dec)
            return rnum

        opt_results.x = round_up(np.atleast_1d(opt_results.x))
        if verbose:
            print(
                f"Optimization completed in time {time.time()-start_time:.2f} seconds."
            )
            print(f"Optimal policy:\t{opt_results.x}")

        ouu_results = {
            "policy": torch.tensor(opt_results.x),
            "OptResults": opt_results,
        }

        # Check optimize results and provide appropriate warnings
        if not opt_results["success"]:
            if np.any(opt_results.x - u_min < 0) or np.any(u_max - opt_results.x < 0):
                warnings.warn(
                    "Optimal intervention policy is out of bounds. Try (i) expanding the bounds_interventions and/or"
                    "(ii) different initial_guess_interventions."
                )
            if opt_results["lowest_optimization_result"]["maxcv"] > 0:
                warnings.warn(
                    "Optimal intervention policy does not satisfy constraints."
                    "Check if the risk_bounds value is appropriate for given problem."
                    "Otherwise, try (i) different initial_guess_interventions, (ii) increasing maxiter/maxfeval,"
                    "and/or (iii) increase n_samples_ouu to improve accuracy of Monte Carlo risk estimation. "
                )

        return ouu_results
