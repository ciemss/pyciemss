from __future__ import annotations

import functools
from typing import Callable, Dict, Optional, Tuple, TypeVar, Union

import mira
import mira.metamodel
import mira.modeling
import mira.sources
import mira.sources.amr
import pyro
import torch
from chirho.dynamical.handlers.solver import Solver, TorchDiffEq
from chirho.dynamical.ops import State, simulate

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

        return simulate(
            self.diff, self.initial_state(), start_time, end_time, solver=solver
        )

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
        setattr(model, "numeric_deriv_func", _compile_deriv(src))
        setattr(model, "numeric_initial_state_func", _compile_initial_state(src))
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
def _compile_deriv(src) -> Callable[..., Tuple[torch.Tensor]]:
    raise NotImplementedError


@functools.singledispatch
def _compile_rate_law(transition) -> Callable[..., Tuple[torch.Tensor]]:
    raise NotImplementedError


@functools.singledispatch
def _compile_initial_state(src) -> Callable[..., Tuple[torch.Tensor]]:
    raise NotImplementedError


@functools.singledispatch
def default_param_values(
    src,
) -> Dict[str, Union[torch.Tensor, pyro.nn.PyroParam, pyro.nn.PyroSample]]:
    raise NotImplementedError


@functools.singledispatch
def default_initial_state(src) -> Optional[State[torch.Tensor]]:
    raise NotImplementedError


@get_name.register
def _get_name_str(name: str) -> str:
    return name
