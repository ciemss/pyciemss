import pyro
import torch

import mira

import pandas as pd

from pyro.infer import Predictive
from pyro import poutine

from pyciemss.interfaces import (
    setup_model,
    reset_model,
    intervene,
    sample,
    calibrate,
    optimize,
    DynamicalSystem,
)

from pyciemss.PetriNetODE.base import get_name
from pyciemss.PetriNetODE.interfaces import load_petri_model

from pyciemss.Ensemble.base import EnsembleSystem, ScaledBetaNoiseEnsembleSystem
from pyciemss.utils.interface_utils import convert_to_output_format, csv_to_list

from typing import Iterable, Optional, Tuple, Callable, Union
import copy

# TODO: probably refactor this out later.
from pyciemss.PetriNetODE.events import (
    StartEvent,
    ObservationEvent,
    LoggingEvent,
    StaticParameterInterventionEvent,
)

EnsembleSolution = Iterable[dict[str, torch.Tensor]]
EnsembleInferredParameters = pyro.nn.PyroModule


def load_and_sample_petri_ensemble(
    petri_model_or_paths: Iterable[
        Union[str, mira.metamodel.TemplateModel, mira.modeling.Model]
    ],
    weights: Iterable[float],
    solution_mappings: Iterable[Callable],
    num_samples: int,
    timepoints: Iterable[float],
    *,
    start_states: Optional[Iterable[dict[str, float]]] = None,
    total_population: float = 1.0,
    pseudocount: float = 1.0,
    dirichlet_concentration: float = 1.0,
    start_time: float = -1e-10,
    method="dopri5",
) -> pd.DataFrame:
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
            - Each element of the iterable is a function that takes in a model output and returns a dict of the form {variable_name: value}.
            - The order of the functions should match the order of the models.
        num_samples: int
            - The number of samples to draw from the model.
        timepoints: [Iterable[float]]
            - The timepoints to simulate the model from. Backcasting and/or forecasting is reflected in the choice of timepoints.
        start_states: Optional[Iterable[dict[str, float]]]
            - Each element of the iterable is the initial state of the component model.
            - If None, the initial state is taken from each of the mira models.
            - Note: Currently users must specify the initial state for all or none of the models.
        total_population: float > 0.0
            - The total population of the model. This is used to scale the model to the correct population.
        pseudocount: float > 0.0
            - The pseudocount to use for adding uncertainty to the model parameters.
            - Larger values of pseudocount correspond to more certainty about the model parameters.
        dirichlet_concentration: float > 0.0
            - The concentration parameter for the dirichlet distribution used to sample the ensemble mixture weights.
            - Larger values of dirichlet_concentration correspond to more certainty about the weights.
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
    models = [
        load_petri_model(
            petri_model_or_path=pmop,
            add_uncertainty=True,
            pseudocount=pseudocount,
        )
        for pmop in petri_model_or_paths
    ]

    # If the user doesn't override the start state, use the initial values from the model.
    if start_states is None:
        start_states = [
            {get_name(v): v.data["initial_value"] for v in model.G.variables.values()}
            for model in models
        ]

    models = setup_model(
        models,
        weights,
        solution_mappings,
        start_time,
        start_states,
        total_population,
        pseudocount,
        dirichlet_concentration,
    )

    samples = sample(
        models,
        timepoints,
        num_samples,
        method=method,
    )
    processed_samples = convert_to_output_format(samples, timepoints)

    return processed_samples

def load_and_calibrate_and_sample_ensemble_model(
    petri_model_or_paths: Iterable[
        Union[str, mira.metamodel.TemplateModel, mira.modeling.Model]
    ],
    data_path: str,
    weights: Iterable[float],
    solution_mappings: Iterable[Callable],
    num_samples: int,
    timepoints: Iterable[float],
    *,
    start_states: Optional[Iterable[dict[str, float]]] = None,
    total_population: float = 1.0,
    pseudocount: float = 1.0,
    dirichlet_concentration: float = 1.0,
    start_time: float = -1e-10,
    num_iterations: int = 1000,
    lr: float = 0.03,
    verbose: bool = False,
    verbose_every: int = 25,
    num_particles: int = 1,
    autoguide=pyro.infer.autoguide.AutoLowRankMultivariateNormal,
    method="dopri5",
) -> pd.DataFrame:
    """
    Load a collection petri net from a file, compile them into an ensemble probabilistic program, calibrate it on data,
    and sample from the calibrated model.

    Args:
        petri_model_or_paths: Iterable[Union[str, mira.metamodel.TemplateModel, mira.modeling.Model]
            - Each element of the iterable is a path to a petri net file, or a petri net object.
            - This path can be a URL or a local path to a mira model or AMR model.
            - Alternatively, this can be a mira template model directly.
        data_path: str
            - The path to the data to calibrate the model to. See notebook/integration_demo/data.csv for an example of the format.
            - The data should be a csv with one column for "time" and remaining columns for each state variable.
            - Each state variable must exactly align with the state variables in the shared ensemble representation. (See `solution_mappings` for more details.)
        weights: Iterable[float]
            - Weights representing prior belief about which models are more likely to be correct.
            - By convention these weights should sum to 1.0.
        solution_mappings: Iterable[Callable]
            - A list of functions that map the output of the model to the output of the shared state space.
            - Each element of the iterable is a function that takes in a model output and returns a dict of the form {variable_name: value}.
            - The order of the functions should match the order of the models.
        num_samples: int
            - The number of samples to draw from the model.
        timepoints: [Iterable[float]]
            - The timepoints to simulate the model from. Backcasting and/or forecasting is reflected in the choice of timepoints.
        start_states: Optional[Iterable[dict[str, float]]]
            - Each element of the iterable is the initial state of the component model.
            - If None, the initial state is taken from each of the mira models.
            - Note: Currently users must specify the initial state for all or none of the models.
        total_population: float > 0.0
            - The total population of the model. This is used to scale the model to the correct population.
        pseudocount: float > 0.0
            - The pseudocount to use for adding uncertainty to the model parameters.
            - Larger values of pseudocount correspond to more certainty about the model parameters.
        dirichlet_concentration: float > 0.0
            - The concentration parameter for the dirichlet distribution used to sample the ensemble mixture weights.
            - Larger values of dirichlet_concentration correspond to more certainty about the weights.
        start_time: float
            - The start time of the model. This is used to align the `start_state` with the `timepoints`.
            - By default we set the `start_time` to be a small negative number to avoid numerical issues w/ collision with the `timepoints` which typically start at 0.
        num_iterations: int
            - The number of iterations to run the calibration for.
        lr: float
            - The learning rate to use for the calibration.
        verbose: bool
            - Whether to print out the calibration progress. This will include summaries of the evidence lower bound (ELBO) and the parameters.
        verbose_every: int
            - How often to print out the loss during calibration.
        num_particles: int
            - The number of particles to use for the calibration. Increasing this value will result in lower variance gradient estimates, but will also increase the computational cost per gradient step.
        autoguide: pyro.infer.autoguide.AutoGuide
            - The autoguide to use for the calibration.
        method: str
            - The method to use for the ODE solver. See `torchdiffeq.odeint` for more details.
            - If performance is incredibly slow, we suggest using `euler` to debug. If using `euler` results in faster simulation, the issue is likely that the model is stiff.
    Returns:
        samples: pd.DataFrame
            - A dataframe containing the samples from the calibrated model.
            
    """


    data = csv_to_list(data_path)

    models = [
        load_petri_model(
            petri_model_or_path=pmop,
            add_uncertainty=True,
            pseudocount=pseudocount,
        )
        for pmop in petri_model_or_paths
    ]

    # If the user doesn't override the start state, use the initial values from the model.
    if start_states is None:
        start_states = [
            {get_name(v): v.data["initial_value"] for v in model.G.variables.values()}
            for model in models
        ]

    models = setup_model(
        models,
        weights,
        solution_mappings,
        start_time,
        start_states,
        total_population,
        pseudocount,
        dirichlet_concentration,
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

    processed_samples = convert_to_output_format(samples, timepoints)

    return processed_samples
    

##############################################################################
# Internal Interfaces Below - TA4 above


# TODO: create better type hint for `models`. Struggled with `Iterable[DynamicalSystem]`.
@setup_model.register(list)
def setup_ensemble_model(
    models: list[DynamicalSystem],
    weights: Iterable[float],
    solution_mappings: Iterable[Callable],
    start_time: float,
    start_states: Iterable[dict[str, float]],
    total_population: float = 1.0,
    noise_pseudocount: float = 1.0,
    dirichlet_concentration: float = 1.0,
) -> EnsembleSystem:
    """
    Instatiate a model for a particular configuration of initial conditions
    """
    ensemble_model = copy.deepcopy(
        ScaledBetaNoiseEnsembleSystem(
            models,
            torch.as_tensor(weights) * dirichlet_concentration,
            solution_mappings,
            total_population,
            noise_pseudocount,
        )
    )
    for i, m in enumerate(ensemble_model.models):
        start_event = StartEvent(start_time, start_states[i])
        m.load_event(start_event)
    return ensemble_model


@reset_model.register
def reset_ensemble_model(ensemble: EnsembleSystem) -> EnsembleSystem:
    """
    Reset a model to its initial state.
    reset_model * setup_model = id
    """
    raise NotImplementedError


@intervene.register
def intervene_ensemble_model(
    ensemble: EnsembleSystem, interventions: Iterable[Tuple[float, str, float]]
) -> EnsembleSystem:
    """
    Intervene on a model.
    """
    raise NotImplementedError


@calibrate.register
def calibrate_ensemble_model(
    ensemble: EnsembleSystem,
    data: Iterable[Tuple[float, dict[str, float]]],
    num_iterations: int = 1000,
    lr: float = 0.03,
    verbose: bool = False,
    verbose_every: int = 25,
    num_particles: int = 1,
    autoguide=pyro.infer.autoguide.AutoLowRankMultivariateNormal,
    method="dopri5",
) -> EnsembleInferredParameters:
    """
    Calibrate a model. Dispatches to the calibrate method of the underlying model.
    This method is only implemented for petri net models.
    """
    # TODO: Refactor the codebase so that this can be implemented for any model that has a calibrate method.
    # This will require pulling out functions for checking the validity of the data, and for setting up the model.

    new_ensemble = copy.deepcopy(ensemble)
    observations = [
        ObservationEvent(timepoint, observation) for timepoint, observation in data
    ]

    # Again, here we assume that all observations are scaled to the first model in the ensemble.
    test_petri = new_ensemble.models[0]

    for obs in observations:
        s = 0.0
        for v in obs.observation.values():
            s += v
            assert 0 <= v <= test_petri.total_population
        assert s <= test_petri.total_population or torch.isclose(
            s, test_petri.total_population
        )

    new_ensemble.load_events(observations)

    guide = autoguide(new_ensemble)
    optim = pyro.optim.Adam({"lr": lr})
    loss = pyro.infer.Trace_ELBO(num_particles=num_particles)
    svi = pyro.infer.SVI(new_ensemble, guide, optim, loss=loss)

    pyro.clear_param_store()

    for i in range(num_iterations):
        loss = svi.step(method=method)
        if verbose:
            if i % verbose_every == 0:
                print(f"iteration {i}: loss = {loss}")

    return guide


@sample.register
def sample_ensemble_model(
    ensemble: EnsembleSystem,
    timepoints: Iterable[float],
    num_samples: int,
    inferred_parameters: Optional[EnsembleInferredParameters] = None,
    *args,
    **kwargs,
) -> EnsembleSolution:
    """
    Sample from an ensemble model.
    """

    logging_events = [LoggingEvent(timepoint) for timepoint in timepoints]
    new_ensemble = copy.deepcopy(ensemble)
    new_ensemble.load_events(logging_events)
    # **kwargs is used to pass in optional model parameters, such as the solver method for an ODE.
    return Predictive(new_ensemble, guide=inferred_parameters, num_samples=num_samples)(
        *args, **kwargs
    )
