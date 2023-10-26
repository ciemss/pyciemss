from __future__ import annotations

import functools
from typing import Callable, Dict, Tuple, TypeVar, Union

import mira
import mira.metamodel
import mira.modeling
import mira.sources
import mira.sources.amr
import pickle
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
        for k, v in _compile_param_values(src).items():
            if hasattr(self, get_name(k)):
                continue

            if isinstance(
                v, (torch.nn.Parameter, pyro.nn.PyroParam, pyro.nn.PyroSample)
            ):
                setattr(self, get_name(k), v)
            elif isinstance(v, torch.Tensor):
                self.register_buffer(get_name(k), v)

        # Compile the numeric derivative of the model from the transition rate laws.
        setattr(self, "numeric_deriv_func", _compile_deriv(src))
        setattr(self, "numeric_initial_state_func", _compile_initial_state(src))

    @pyro.nn.pyro_method
    def deriv(self, X: State[torch.Tensor]) -> None:
        return eval_deriv(self.src, self, X)

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
        for k in _compile_param_values(self.src).keys():
            getattr(self, get_name(k))

        return simulate(
            self.deriv, self.initial_state(), start_time, end_time, solver=solver
        )

    def save(self, dest: str) -> None:
        assert dest[-4:] == '.pkl', 'Model must be saved as a .pkl file.'
        with open(dest, "wb") as f:
            pickle.dump(self, f)

    @functools.singledispatchmethod
    @classmethod
    def load(cls, src) -> "CompiledDynamics":
        raise NotImplementedError

    @load.register(str)
    @classmethod
    def _load_from_url_or_path(cls, path: str):
        if "https://" in path:
            model = mira.sources.amr.model_from_url(path)
        elif path[-4:] == '.pkl':
            with open(path, 'rb') as f:
                return pickle.load(f)
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
        return cls(src)


@functools.singledispatch
def _compile_deriv(src) -> Callable[..., Tuple[torch.Tensor]]:
    raise NotImplementedError


@functools.singledispatch
def _compile_initial_state(src) -> Callable[..., Tuple[torch.Tensor]]:
    raise NotImplementedError


@functools.singledispatch
def _compile_param_values(
    src,
) -> Dict[str, Union[torch.Tensor, pyro.nn.PyroParam, pyro.nn.PyroSample]]:
    raise NotImplementedError


@functools.singledispatch
def eval_deriv(src, param_module: pyro.nn.PyroModule, X: State[T]) -> State[T]:
    raise NotImplementedError


@functools.singledispatch
def eval_initial_state(src, param_module: pyro.nn.PyroModule) -> State[T]:
    raise NotImplementedError


@functools.singledispatch
def get_name(obj) -> str:
    raise NotImplementedError


@get_name.register
def _get_name_str(name: str) -> str:
    return name
