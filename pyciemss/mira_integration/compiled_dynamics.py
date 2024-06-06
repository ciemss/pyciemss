from __future__ import annotations

import numbers
from typing import Callable, Dict, Optional, Tuple, TypeVar, Union

import mira
import mira.metamodel
import mira.modeling
import mira.sources
import mira.sources.amr
import numpy
import pyro
import sympy
import sympytorch
import torch
from chirho.dynamical.ops import State

from pyciemss.compiled_dynamics import (
    _compile_deriv,
    _compile_initial_state,
    _compile_observables,
    _compile_param_values,
    eval_deriv,
    eval_initial_state,
    eval_observables,
    get_name,
)
from pyciemss.mira_integration.distributions import mira_distribution_to_pyro

S = TypeVar("S")
T = TypeVar("T")


@_compile_deriv.register(mira.modeling.Model)
def _compile_deriv_mira(src: mira.modeling.Model) -> Callable[..., Tuple[torch.Tensor]]:
    symbolic_deriv = {get_name(var): 0 for var in src.variables.values()}
    for transition in src.transitions.values():
        flux = transition.template.rate_law.args[0]
        for c in transition.consumed:
            symbolic_deriv[get_name(c)] -= flux
        for p in transition.produced:
            symbolic_deriv[get_name(p)] += flux
    simplified_deriv = {k: sympy.simplify(v) for k, v in symbolic_deriv.items()}
    numeric_deriv_func = sympytorch.SymPyModule(
        expressions=[simplified_deriv[get_name(var)] for var in src.variables.values()]
    )
    return numeric_deriv_func


@_compile_initial_state.register(mira.modeling.Model)
def _compile_initial_state_mira(
    src: mira.modeling.Model,
) -> Callable[..., Tuple[torch.Tensor]]:
    symbolic_initials = {
        get_name(var): var.data["expression"].args[0] for var in src.variables.values()
    }
    numeric_initial_state_func = sympytorch.SymPyModule(
        expressions=[symbolic_initials[get_name(var)] for var in src.variables.values()]
    )
    return numeric_initial_state_func


@_compile_observables.register(mira.modeling.Model)
def _compile_observables_mira(
    src: mira.modeling.Model,
) -> Optional[Callable[..., Tuple[torch.Tensor]]]:
    if len(src.observables) == 0:
        return None

    symbolic_observables = {
        get_name(obs): obs.observable.expression.args[0]
        for obs in src.observables.values()
    }

    numeric_observables_func = sympytorch.SymPyModule(
        expressions=[
            symbolic_observables[get_name(obs)] for obs in src.observables.values()
        ]
    )
    return numeric_observables_func


@_compile_param_values.register(mira.modeling.Model)
def _compile_param_values_mira(
    src: mira.modeling.Model,
) -> Dict[str, Union[torch.Tensor, pyro.nn.PyroParam, pyro.nn.PyroSample]]:
    values = {}
    for param_info in src.parameters.values():
        param_name = get_name(param_info)

        if param_info.placeholder:
            continue

        param_dist = getattr(param_info, "distribution", None)
        if param_dist is None:
            param_value = param_info.value
        else:
            param_value = mira_distribution_to_pyro(param_dist)

        if isinstance(param_value, torch.nn.Parameter):
            values[param_name] = pyro.nn.PyroParam(param_value)
        elif isinstance(param_value, pyro.distributions.Distribution):
            values[param_name] = pyro.nn.PyroSample(param_value)
        elif isinstance(param_value, (numbers.Number, numpy.ndarray, torch.Tensor)):
            values[param_name] = torch.as_tensor(param_value)
        else:
            raise TypeError(f"Unknown parameter type: {type(param_value)}")

    return values


@eval_deriv.register(mira.modeling.Model)
def _eval_deriv_mira(
    src: mira.modeling.Model,
    param_module: pyro.nn.PyroModule,
    X: State[torch.Tensor],
) -> State[torch.Tensor]:
    parameters = {
        get_name(param_info): getattr(param_module, get_name(param_info))
        for param_info in src.parameters.values()
        if not param_info.placeholder
    }

    numeric_deriv = param_module.numeric_deriv_func(**X, **parameters)

    dX: State[torch.Tensor] = dict()
    for i, var in enumerate(src.variables.values()):
        k = get_name(var)
        dX[k] = numeric_deriv[..., i]
    return dX


@eval_initial_state.register(mira.modeling.Model)
def _eval_initial_state_mira(
    src: mira.modeling.Model,
    param_module: pyro.nn.PyroModule,
) -> State[torch.Tensor]:
    parameters = {
        get_name(param_info): getattr(param_module, get_name(param_info))
        for param_info in src.parameters.values()
        if not param_info.placeholder
    }

    numeric_initial_state = param_module.numeric_initial_state_func(**parameters)

    X: State[torch.Tensor] = dict()
    for i, var in enumerate(src.variables.values()):
        k = get_name(var)
        X[k] = numeric_initial_state[..., i]
    return X


@eval_observables.register(mira.modeling.Model)
def _eval_observables_mira(
    src: mira.modeling.Model,
    param_module: pyro.nn.PyroModule,
    X: State[torch.Tensor],
) -> State[torch.Tensor]:
    if len(src.observables) == 0:
        return dict()

    parameters = {
        get_name(param_info): getattr(param_module, get_name(param_info))
        for param_info in src.parameters.values()
        if not param_info.placeholder
    }

    # TODO: support event_dim > 0 upstream in ChiRho
    # Default to time being the rightmost dimension
    parameters_expanded = {
        k: torch.unsqueeze(v, -1) if len(v.size()) > 0 else v
        for k, v in parameters.items()
    }

    numeric_observables = param_module.numeric_observables_func(
        **X, **parameters_expanded
    )

    observables: State[torch.Tensor] = dict()
    for i, obs in enumerate(src.observables.values()):
        k = get_name(obs)
        observables[k] = numeric_observables[..., i]

    return observables


@get_name.register
def _get_name_mira_variable(var: mira.modeling.Variable) -> str:
    return var.data["name"]


@get_name.register
def _get_name_mira_transition(trans: mira.modeling.Transition) -> str:
    return f"trans_{trans.key}"


@get_name.register
def _get_name_mira_modelparameter(param: mira.modeling.ModelParameter) -> str:
    return str(param.key)


@get_name.register
def _get_name_mira_model_observable(obs: mira.modeling.ModelObservable) -> str:
    return obs.observable.name


@get_name.register
def _get_name_mira_observable(obs: mira.modeling.Observable) -> str:
    return obs.name
