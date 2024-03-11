import pyro
import pytest
import torch
from chirho.dynamical.handlers.solver import TorchDiffEq

from pyciemss.compiled_dynamics import CompiledDynamics
from pyciemss.interruptions import (
    DynamicParameterIntervention,
    StaticParameterIntervention,
)

from .fixtures import (
    END_TIMES,
    MODELS,
    START_TIMES,
    check_is_state,
    check_states_match,
    check_states_match_in_all_but_values,
)

INTERVENTION_ASSIGNMENTS = [torch.tensor(2.0), lambda x: x * 2.0]
INTERVENTION_HANDLER_TYPES = ["static", "dynamic"]


@pytest.mark.parametrize("model_fixture", MODELS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("intervention_assignment", INTERVENTION_ASSIGNMENTS)
@pytest.mark.parametrize("intervention_handler_type", INTERVENTION_HANDLER_TYPES)
def test_parameter_intervention_before_end(
    model_fixture,
    start_time,
    end_time,
    intervention_assignment,
    intervention_handler_type,
):
    model = CompiledDynamics.load(model_fixture.url)
    assert isinstance(model, CompiledDynamics)

    intervention_time = start_time + (end_time - start_time) / 2.0
    parameter = model_fixture.important_parameter
    intervention = {parameter: intervention_assignment}

    if intervention_handler_type == "static":
        intervention_handler = StaticParameterIntervention(
            time=intervention_time, intervention=intervention
        )
    elif intervention_handler_type == "dynamic":

        def event_fn(time: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            return time - intervention_time

        intervention_handler = DynamicParameterIntervention(
            event_fn=event_fn, intervention=intervention
        )

    if parameter is None:
        pytest.skip("Model does not have an important parameter.")

    with pyro.poutine.seed(rng_seed=0):
        with TorchDiffEq():
            with intervention_handler:
                simulation1 = model(
                    torch.as_tensor(start_time), torch.as_tensor(end_time)
                )

    with pyro.poutine.seed(rng_seed=0):
        with TorchDiffEq():
            simulation2 = model(torch.as_tensor(start_time), torch.as_tensor(end_time))

    check_is_state(simulation1, torch.Tensor)
    check_is_state(simulation2, torch.Tensor)
    check_states_match_in_all_but_values(simulation1, simulation2, state_ndim=0)


@pytest.mark.parametrize("model_fixture", MODELS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("intervention_assignment", INTERVENTION_ASSIGNMENTS)
@pytest.mark.parametrize("intervention_handler_type", INTERVENTION_HANDLER_TYPES)
def test_parameter_intervention_after_end(
    model_fixture,
    start_time,
    end_time,
    intervention_assignment,
    intervention_handler_type,
):
    model = CompiledDynamics.load(model_fixture.url)
    assert isinstance(model, CompiledDynamics)

    intervention_time = end_time + 1.0
    parameter = model_fixture.important_parameter
    intervention = {parameter: intervention_assignment}

    if intervention_handler_type == "static":
        intervention_handler = StaticParameterIntervention(
            time=intervention_time, intervention=intervention
        )
    elif intervention_handler_type == "dynamic":

        def event_fn(time: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            return time - intervention_time

        intervention_handler = DynamicParameterIntervention(
            event_fn=event_fn, intervention=intervention
        )

    if parameter is None:
        pytest.skip("Model does not have an important parameter.")

    with pyro.poutine.seed(rng_seed=0):
        with TorchDiffEq():
            with intervention_handler:
                simulation1 = model(
                    torch.as_tensor(start_time), torch.as_tensor(end_time)
                )

    with pyro.poutine.seed(rng_seed=0):
        with TorchDiffEq():
            simulation2 = model(torch.as_tensor(start_time), torch.as_tensor(end_time))

    check_is_state(simulation1, torch.Tensor)
    check_is_state(simulation2, torch.Tensor)
    check_states_match(simulation1, simulation2)

@pytest.mark.parametrize("model_fixture", MODELS)
@pytest.mark.parametrize("start_time", START_TIMES)
@pytest.mark.parametrize("end_time", END_TIMES)
@pytest.mark.parametrize("intervention_handler_type", INTERVENTION_HANDLER_TYPES)
def test_parameter_intervention_noop(
    model_fixture,
    start_time,
    end_time,
    intervention_handler_type,
):
    model = CompiledDynamics.load(model_fixture.url)
    assert isinstance(model, CompiledDynamics)

    intervention_time = start_time + (end_time - start_time) / 2.0
    parameter = model_fixture.important_parameter
    intervention = {parameter: lambda x : x}

    if intervention_handler_type == "static":
        intervention_handler = StaticParameterIntervention(
            time=intervention_time, intervention=intervention
        )
    elif intervention_handler_type == "dynamic":

        def event_fn(time: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            return time - intervention_time

        intervention_handler = DynamicParameterIntervention(
            event_fn=event_fn, intervention=intervention
        )

    if parameter is None:
        pytest.skip("Model does not have an important parameter.")

    with pyro.poutine.seed(rng_seed=0):
        with TorchDiffEq():
            with intervention_handler:
                simulation1 = model(
                    torch.as_tensor(start_time), torch.as_tensor(end_time)
                )

    with pyro.poutine.seed(rng_seed=0):
        with TorchDiffEq():
            simulation2 = model(torch.as_tensor(start_time), torch.as_tensor(end_time))

    check_is_state(simulation1, torch.Tensor)
    check_is_state(simulation2, torch.Tensor)
    check_states_match(simulation1, simulation2)