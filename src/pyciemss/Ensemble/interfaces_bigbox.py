from typing_extensions import deprecated
import pyro

import mira

from ..interfaces import DEFAULT_QUANTILES
from .interfaces import (
    setup_model,
    sample,
    calibrate,
    prepare_interchange_dictionary,
)

from pyciemss.PetriNetODE.base import get_name
from pyciemss.PetriNetODE.interfaces import load_petri_model

from pyciemss.utils.interface_utils import (
    csv_to_list,
    create_mapping_function_from_observables,
)

from typing import Iterable, Optional, Union
from pyciemss.custom_decorators import pyciemss_logging_wrapper


@deprecated(
    "'Big Box' interfaces were a stop-gap for July 2023 hackathon."
    "Please call the appropriate sequence from 'interfaces' insteads."
)
@pyciemss_logging_wrapper
def load_and_sample_petri_ensemble(
    petri_model_or_paths: Iterable[
        Union[str, mira.metamodel.TemplateModel, mira.modeling.Model]
    ],
    weights: Iterable[float],
    solution_mappings: Iterable[dict[str, str]],
    num_samples: int,
    timepoints: Iterable[float],
    *,
    start_states: Optional[Iterable[dict[str, float]]] = None,
    total_population: float = 1.0,
    dirichlet_concentration: float = 1.0,
    start_time: float = -1e-10,
    method="dopri5",
    compile_rate_law_p: bool = True,
    time_unit: Optional[str] = None,
    visual_options: Union[None, bool, dict[str, any]] = None,
    alpha_qs: Optional[Iterable[float]] = DEFAULT_QUANTILES,
    stacking_order: Optional[str] = "timepoints",
) -> dict:
    """
    Load a petri net from a file, compile it into a probabilistic program, and sample from it.

    Args:
        petri_model_or_paths: Iterable[Union[str, mira.metamodel.TemplateModel, mira.modeling.Model]
            - Each element of the iterable is a path to a petri net file, or a petri net object.
            - This path can be a URL or a local path to a mira model or AMR model.
            - Alternatively, this can be a mira template model directly.
        weights: Iterable[float]
            - Weights representing prior belief about which models are more likely to be correct.
            - By convention these weights should sum to 1.0.
        solution_mappings: Iterable[Callable]
            - A list of functions that map the output of the model to the output of the shared state space.
            - Each element of the iterable is a function that takes in a model output
              and returns a dict of the form {variable_name: value}.
            - The order of the functions should match the order of the models.
        num_samples: int
            - The number of samples to draw from the model.
        timepoints: [Iterable[float]]
            - The timepoints to simulate the model from. Backcasting and/or forecasting is reflected
                in the choice of timepoints.
        start_states: Optional[Iterable[dict[str, float]]]
            - Each element of the iterable is the initial state of the component model.
            - If None, the initial state is taken from each of the mira models.
            - Note: Currently users must specify the initial state for all or none of the models.
        total_population: float > 0.0
            - The total population of the model. This is used to scale the model to the correct population.
        pseudocount: float > 0.0
            - The pseudocount to use for adding uncertainty to the observations.
            - Larger values of pseudocount correspond to more certainty about the observations.
        dirichlet_concentration: float > 0.0
            - The concentration parameter for the dirichlet distribution used to sample the ensemble mixture weights.
            - Larger values of dirichlet_concentration correspond to more certainty about the weights.
        start_time: float
            - The start time of the model. This is used to align the `start_state` with the `timepoints`.
            - By default we set the `start_time` to be a small negative number to avoid numerical
              issues w/ collision with the `timepoints` which typically start at 0.
        method: str
            - The method to use for solving the ODE. See torchdiffeq's `odeint` method for more details.
            - If performance is incredibly slow, we suggest using `euler` to debug. If using `euler` results
              in faster simulation, the issue is likely that the model is stiff.
        time_unit: str
            - Time unit (used for labeling outputs)
        visual_options: None, bool, dict[str, any]
            - True output a visual
            - False do not output a visual
            - dict output a visual with the dictionary passed to the visualization as kwargs
        alpha_qs: Optional[Iterable[float]]
            - The quantiles required for estimating weighted interval score to test ensemble forecasting accuracy.
        stacking_order: Optional[str]
            - The stacking order requested for the ensemble quantiles to keep the selected quantity
              together for each state.
            - Options: "timepoints" or "quantiles"

    Returns:
        result: dict
            - Dictionary of outputs with following attribute:
                * data: The samples from the model as a pandas DataFrame.
                * quantiles: The quantiles for ensemble score calculation as a pandas DataFrames.
                * visual: Visualization. (If visual_options is truthy)
    """
    models = [
        load_petri_model(
            petri_model_or_path=pmop,
            add_uncertainty=False,
            compile_rate_law_p=compile_rate_law_p,
            compile_observables_p=True,
        )
        for pmop in petri_model_or_paths
    ]

    solution_mapping_fs = []

    for i, model in enumerate(models):
        solution_mapping_f = create_mapping_function_from_observables(
            model, solution_mappings[i]
        )
        solution_mapping_fs.append(solution_mapping_f)

    # If the user doesn't override the start state, use the initial values from the model.
    if start_states is None:
        start_states = [
            {get_name(v): v.data["initial_value"] for v in model.G.variables.values()}
            for model in models
        ]

    models = setup_model(
        models,
        weights,
        solution_mapping_fs,
        start_time,
        start_states,
        total_population=total_population,
        dirichlet_concentration=dirichlet_concentration,
    )

    samples = sample(
        models,
        timepoints,
        num_samples,
        method=method,
    )

    return prepare_interchange_dictionary(
        samples, timepoints, time_unit, alpha_qs, stacking_order, visual_options
    )


@deprecated(
    "'Big Box' interfaces were a stop-gap for July 2023 hackathon."
    "Please call the appropriate sequence from 'interfaces' insteads."
)
@pyciemss_logging_wrapper
def load_and_calibrate_and_sample_ensemble_model(
    petri_model_or_paths: Iterable[
        Union[str, mira.metamodel.TemplateModel, mira.modeling.Model]
    ],
    data_path: str,
    weights: Iterable[float],
    solution_mappings: Iterable[dict[str, str]],
    num_samples: int,
    timepoints: Iterable[float],
    *,
    start_states: Optional[Iterable[dict[str, float]]] = None,
    total_population: float = 1.0,
    noise_model: str = "scaled_normal",
    noise_scale: float = 0.1,
    dirichlet_concentration: float = 1.0,
    start_time: float = -1e-10,
    num_iterations: int = 1000,
    lr: float = 0.03,
    verbose: bool = False,
    verbose_every: int = 25,
    num_particles: int = 1,
    autoguide=pyro.infer.autoguide.AutoLowRankMultivariateNormal,
    compile_rate_law_p: bool = True,
    method="dopri5",
    time_unit: Optional[str] = None,
    visual_options: Union[None, bool, dict[str, any]] = None,
    alpha_qs: Optional[Iterable[float]] = DEFAULT_QUANTILES,
    stacking_order: Optional[str] = "timepoints",
) -> dict:
    """
    Load a collection petri net from a file, compile them into an ensemble probabilistic program, calibrate it on data,
    and sample from the calibrated model.

    Args:
        petri_model_or_paths: Iterable[Union[str, mira.metamodel.TemplateModel, mira.modeling.Model]
            - Each element of the iterable is a path to a petri net file, or a petri net object.
            - This path can be a URL or a local path to a mira model or AMR model.
            - Alternatively, this can be a mira template model directly.
        data_path: str
            - The path to the data to calibrate the model to. See notebook/integration_demo/data.csv
              for an example of the format.
            - The data should be a csv with one column for "time" and remaining columns for each state variable.
            - Each state variable must exactly align with the state variables in the shared ensemble representation.
              (See `solution_mappings` for more details.)
        weights: Iterable[float]
            - Weights representing prior belief about which models are more likely to be correct.
            - By convention these weights should sum to 1.0.
        solution_mappings: Iterable[Callable]
            - A list of functions that map the output of the model to the output of the shared state space.
            - Each element of the iterable is a function that takes in a model output and returns a dict of
              the form {variable_name: value}.
            - The order of the functions should match the order of the models.
        num_samples: int
            - The number of samples to draw from the model.
        timepoints: [Iterable[float]]
            - The timepoints to simulate the model from. Backcasting and/or forecasting is reflected
              in the choice of timepoints.
        start_states: Optional[Iterable[dict[str, float]]]
            - Each element of the iterable is the initial state of the component model.
            - If None, the initial state is taken from each of the mira models.
            - Note: Currently users must specify the initial state for all or none of the models.
        total_population: float > 0.0
            - The total population of the model. This is used to scale the model to the correct population.
        pseudocount: float > 0.0
            - The pseudocount to use for adding uncertainty to the observations.
            - Larger values of pseudocount correspond to more certainty about the observations.
        dirichlet_concentration: float > 0.0
            - The concentration parameter for the dirichlet distribution used to sample the ensemble mixture weights.
            - Larger values of dirichlet_concentration correspond to more certainty about the weights.
        start_time: float
            - The start time of the model. This is used to align the `start_state` with the `timepoints`.
            - By default we set the `start_time` to be a small negative number to avoid numerical issues
              w/ collision with the `timepoints` which typically start at 0.
        num_iterations: int
            - The number of iterations to run the calibration for.
        lr: float
            - The learning rate to use for the calibration.
        verbose: bool
            - Whether to print out the calibration progress. This will include summaries of the evidence lower
              bound (ELBO) and the parameters.
        verbose_every: int
            - How often to print out the loss during calibration.
        num_particles: int
            - The number of particles to use for the calibration. Increasing this value will result in lower variance
              gradient estimates, but will also increase the computational cost per gradient step.
        autoguide: pyro.infer.autoguide.AutoGuide
            - The autoguide to use for the calibration.
        method: str
            - The method to use for the ODE solver. See `torchdiffeq.odeint` for more details.
            - If performance is incredibly slow, we suggest using `euler` to debug. If using `euler` results
              in faster simulation, the issue is likely that the model is stiff.
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
                * data: The samples from the calibrated model as a pandas DataFrame.
                * quantiles: The quantiles for ensemble score calculation after calibration as a pandas DataFrames.
                * visual: Visualization. (If visual_options is truthy)
    """

    data = csv_to_list(data_path)

    models = [
        load_petri_model(
            petri_model_or_path=pmop,
            add_uncertainty=False,
            compile_rate_law_p=compile_rate_law_p,
            compile_observables_p=True,
        )
        for pmop in petri_model_or_paths
    ]

    # If the user doesn't override the start state, use the initial values from the model.
    if start_states is None:
        start_states = [
            {get_name(v): v.data["initial_value"] for v in model.G.variables.values()}
            for model in models
        ]

    solution_mapping_fs = []

    for i, model in enumerate(models):
        solution_mapping_f = create_mapping_function_from_observables(
            model, solution_mappings[i]
        )
        solution_mapping_fs.append(solution_mapping_f)

    models = setup_model(
        models,
        weights,
        solution_mapping_fs,
        start_time,
        start_states,
        total_population=total_population,
        noise_model=noise_model,
        noise_scale=noise_scale,
        dirichlet_concentration=dirichlet_concentration,
    )

    inferred_parameters = calibrate(
        models,
        data,
        num_iterations,
        lr,
        verbose,
        verbose_every,
        num_particles,
        autoguide,
        method=method,
    )

    samples = sample(
        models,
        timepoints,
        num_samples,
        inferred_parameters=inferred_parameters,
        method=method,
    )

    return prepare_interchange_dictionary(
        samples, timepoints, time_unit, alpha_qs, stacking_order, visual_options
    )
