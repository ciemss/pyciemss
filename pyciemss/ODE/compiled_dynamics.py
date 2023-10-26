from __future__ import annotations

import functools
import json
import numbers
import os
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
from chirho.dynamical.handlers.solver import Solver, TorchDiffEq
from chirho.dynamical.ops import State, simulate

from pyciemss.mira_utils.distributions import mira_distribution_to_pyro

S = TypeVar("S")
T = TypeVar("T")


class CompiledDynamics(pyro.nn.PyroModule):
    def __init__(self, src, **kwargs):
        super().__init__()
        self.src = src
        for k, v in default_param_values(src).items():
            if hasattr(self, get_name(k)):
                continue

            if isinstance(
                v, (torch.nn.Parameter, pyro.nn.PyroParam, pyro.nn.PyroSample)
            ):
                setattr(self, get_name(k), v)
            elif isinstance(v, torch.Tensor):
                self.register_buffer(get_name(k), v)

    @pyro.nn.pyro_method
    def diff(self, X: State[torch.Tensor]) -> None:
        return eval_diff(self.src, self, X)
    
    @pyro.nn.pyro_method
    def initial_state(self) -> State[torch.Tensor]:
        return eval_initial_state(self.src, self)

    def forward(
        self,
        start_time: torch.Tensor,
        end_time: torch.Tensor,
        solver: Solver = TorchDiffEq(),
    ):
        # Initialize random parameters once before simulating.
        # This is necessary because the parameters are PyroSample objects.
        for k in default_param_values(self.src).keys():
            getattr(self, get_name(k))

        return simulate(self.diff, self.initial_state(), start_time, end_time, solver=solver)

    @functools.singledispatchmethod
    @classmethod
    def load(cls, src) -> "CompiledDynamics":
        raise NotImplementedError

    @load.register(str)
    @classmethod
    def _load_from_url_or_path(cls, path: str):
        if "https://" in path:
            model = mira.sources.amr.model_from_url(path)
        else:
            model = mira.sources.amr.model_from_json_file(path)
        return cls.load(model)

    @load.register(dict)
    @classmethod
    def _load_from_json(cls, model_json: dict):
        return cls.load(mira.sources.amr.model_from_json(model_json))

    @load.register(mira.metamodel.TemplateModel)
    @classmethod
    def _load_from_template_model(cls, template: mira.metamodel.TemplateModel):
        model = cls.load(mira.modeling.Model(template))

        # Check if all parameter names are strings
        if all(isinstance(param.key, str) for param in model.src.parameters.values()):
            return model
        else:
            new_template = mira.metamodel.ops.aggregate_parameters(template)
            return cls.load(mira.modeling.Model(new_template))
        
    @load.register(mira.modeling.Model)
    @classmethod
    def _load_from_mira_model(cls, src: mira.modeling.Model):
        model = cls(src)
        # Compile the numeric derivative of the model from the transition rate laws.
        setattr(model, "numeric_deriv", _compile_deriv(src))
        setattr(model, "numeric_initial_state", _compile_initial_state(src))
        for trans in src.transitions.values():
            # These are not used in the compiled model, but are helpful for inspecting the rate laws.
            setattr(model, f"rate_law_{get_name(trans)}", _compile_rate_law(trans))

        return model


@functools.singledispatch
def eval_diff(src, param_module: pyro.nn.PyroModule, X: State[T]) -> State[T]:
    raise NotImplementedError


@functools.singledispatch
def eval_initial_state(src, param_module: pyro.nn.PyroModule) -> State[T]:
    raise NotImplementedError


@functools.singledispatch
def get_name(obj) -> str:
    raise NotImplementedError


@functools.singledispatch
def default_param_values(
    src,
) -> Dict[str, Union[torch.Tensor, pyro.nn.PyroParam, pyro.nn.PyroSample]]:
    raise NotImplementedError


@functools.singledispatch
def default_initial_state(src) -> Optional[State[torch.Tensor]]:
    raise NotImplementedError


def _compile_deriv(src: mira.modeling.Model) -> Callable[..., Tuple[torch.Tensor]]:
    symbolic_deriv = {get_name(var): 0 for var in src.variables.values()}
    for transition in src.transitions.values():
        flux = transition.template.rate_law.args[0]
        for c in transition.consumed:
            symbolic_deriv[get_name(c)] -= flux
        for p in transition.produced:
            symbolic_deriv[get_name(p)] += flux
    simplified_deriv = {k: sympy.simplify(v) for k, v in symbolic_deriv.items()}
    numeric_deriv = sympytorch.SymPyModule(
        expressions=[simplified_deriv[get_name(var)] for var in src.variables.values()]
    )
    return numeric_deriv


def _compile_rate_law(
    transition: mira.modeling.Transition,
) -> Callable[..., Tuple[torch.Tensor]]:
    return sympytorch.SymPyModule(expressions=[transition.template.rate_law.args[0]])

def _compile_initial_state(
    src: mira.modeling.Model,
) -> Callable[..., Tuple[torch.Tensor]]:
    symbolic_initials = {get_name(var): src.template_model.initials[get_name(var)].expression.args[0] for var in src.variables.values()}
    return sympytorch.SymPyModule(expressions=[symbolic_initials[get_name(var)] for var in src.variables.values()])

@eval_diff.register(mira.modeling.Model)
def _eval_diff_compiled_mira(
    src: mira.modeling.Model,
    param_module: pyro.nn.PyroModule,
    X: State[torch.Tensor],
) -> State[torch.Tensor]:
    dX = State()

    parameters = {
        get_name(param_info): getattr(param_module, get_name(param_info))
        for param_info in src.parameters.values()
    }

    numeric_deriv = param_module.numeric_deriv(**X, **parameters)
    for i, var in enumerate(src.variables.values()):
        k = get_name(var)
        dX[k] = numeric_deriv[i]
    return dX

@eval_initial_state.register(mira.modeling.Model)
def _eval_initial_state_compiled_mira(
    src: mira.modeling.Model,
    param_module: pyro.nn.PyroModule,
) -> State[torch.Tensor]:
    parameters = {
        get_name(param_info): getattr(param_module, get_name(param_info))
        for param_info in src.parameters.values()
    }

    numeric_result = param_module.numeric_initial_state(**parameters)

    X = State()
    for i, var in enumerate(src.variables.values()):
        k = get_name(var)
        X[k] = numeric_result[i]
    return X

@get_name.register
def _get_name_str(name: str) -> str:
    return name


@get_name.register
def _get_name_variable(var: mira.modeling.Variable) -> str:
    return var.data["name"]


@get_name.register
def _get_name_transition(trans: mira.modeling.Transition) -> str:
    return f"trans_{trans.key}"


@get_name.register
def _get_name_modelparameter(param: mira.modeling.ModelParameter) -> str:
    return str(param.key)


@default_param_values.register(mira.modeling.Model)
def _mira_default_param_values(
    src: mira.modeling.Model,
) -> Dict[str, Union[torch.Tensor, pyro.nn.PyroParam, pyro.nn.PyroSample]]:
    values = {}
    for param_info in src.parameters.values():
        param_name = get_name(param_info)

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
