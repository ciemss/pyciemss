from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union

import mira
import mira.metamodel
import mira.modeling
import mira.sources
import mira.sources.amr
import pyro
import torch
from askem_model_representations import model_inventory
from chirho.dynamical.handlers import LogTrajectory
from chirho.dynamical.ops import State, simulate

S = TypeVar("S")
T = TypeVar("T")

RAISE_ON_MODEL_CHECK_ERROR = True


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
                setattr(self, f"persistent_{get_name(k)}", v)
            elif isinstance(v, torch.Tensor):
                self.register_buffer(f"persistent_{get_name(k)}", v)

        # Compile the numeric derivative of the model from the transition rate laws.
        setattr(self, "numeric_deriv_func", _compile_deriv(src))
        setattr(self, "numeric_initial_state_func", _compile_initial_state(src))
        setattr(self, "numeric_observables_func", _compile_observables(src))

        self.instantiate_parameters()

    @pyro.nn.pyro_method
    def deriv(self, X: State[torch.Tensor]) -> None:
        return eval_deriv(self.src, self, X)

    @pyro.nn.pyro_method
    def initial_state(self) -> State[torch.Tensor]:
        return eval_initial_state(self.src, self)

    @pyro.nn.pyro_method
    def observables(self, X: State[torch.Tensor]) -> State[torch.Tensor]:
        return eval_observables(self.src, self, X)

    @pyro.nn.pyro_method
    def instantiate_parameters(self):
        # Initialize random parameters once before simulating.
        # This is necessary because the parameters are PyroSample objects.
        for k in _compile_param_values(self.src).keys():
            param_name = get_name(k)
            # Separating the persistent parameters from the non-persistent ones
            # is necessary because the persistent parameters are PyroSample objects representing the distribution,
            # and should not be modified during intervention.
            param_val = getattr(self, f"persistent_{param_name}")
            self.register_buffer(get_name(k), param_val)

    def forward(
        self,
        start_time: torch.Tensor,
        end_time: torch.Tensor,
        logging_times: Optional[torch.Tensor] = None,
        is_traced: bool = False,
    ):
        self.instantiate_parameters()

        if logging_times is not None:
            with LogTrajectory(logging_times) as lt:
                simulate(self.deriv, self.initial_state(), start_time, end_time)
                state = lt.trajectory
        else:
            state = simulate(self.deriv, self.initial_state(), start_time, end_time)

        observables = self.observables(state)

        if is_traced:
            # Add the observables to the trace so that they can be accessed later.
            [
                pyro.deterministic(f"{name}_state", value)
                for name, value in state.items()
            ]
            [
                pyro.deterministic(f"{name}_observable", value)
                for name, value in observables.items()
            ]

        return {**state, **observables}

    @functools.singledispatchmethod
    @classmethod
    def load(cls, src, *, checks={}) -> "CompiledDynamics":
        raise NotImplementedError

    @load.register(str)
    @classmethod
    def _load_from_url_or_path(cls, path: str, *, checks={}):
        cls.check_model(path, checks)

        if "https://" in path:
            model = mira.sources.amr.model_from_url(path)
        else:
            model = mira.sources.amr.model_from_json_file(path)

        model = cls.load(model)
        return model

    @load.register(dict)
    @classmethod
    def _load_from_json(cls, model_json: dict, *, checks={}):
        cls.check_model(model_json, checks)
        model = cls.load(mira.sources.amr.model_from_json(model_json))
        return model

    @load.register(mira.metamodel.TemplateModel)
    @classmethod
    def _load_from_template_model(
        cls, template: mira.metamodel.TemplateModel, *, checks={}
    ):
        model = cls.load(mira.modeling.Model(template))
        # TODO: We cannot directly check mira models, only the JSONs.
        #      Should we reformulate model-inventory to load mira instead?
        # check_model(model)
        return model

    @load.register(mira.modeling.Model)
    @classmethod
    def _load_from_mira_model(cls, src: mira.modeling.Model, *, checks={}):
        model = cls(src)
        # TODO: We cannot directly check mira models, only the JSONs.
        #      Should we reformulate model-inventory to load mira instead?
        # check_model(model)
        return model

    @classmethod
    def check_model(cls, model: Dict, checks: Dict[str, Tuple[Any, str]]):
        """
        Check a model for 'goodness'.  If it 'bad', raise/warn a descriptive error.

        Will raise an exception on RAISE_ON_MODEL_CHECK_ERROR is True, or warn if RAISE_ON_MODEL_CHECK_ERROR is False.
        checks -- The key is th expected key in an askem_model_representations.model_inventory result.
                 The value is a pair of expected-value  and a message if the expected value is not found.

        """

        def _warn(msg):
            warnings.warn(msg)

        def _raise(msg):
            raise ValueError(msg)

        on_issue = _raise if RAISE_ON_MODEL_CHECK_ERROR else _warn

        # TODO: To provide more descriptive error messages, don't use "summary"
        #      Then we can pipe the details into the message.
        #      For this, remove must_be_X and make it checks: Dict[str, Callable]
        inventory = model_inventory.check_amr(model, summary=True)

        for key, (value, message) in checks.items():
            if key not in inventory:
                on_issue(
                    f"Malformed model inventory for requested checks. Could not find '{key}' in {inventory}"
                )

            if inventory[key] != value:
                on_issue(message)


@functools.singledispatch
def _compile_deriv(src) -> Callable[..., Tuple[torch.Tensor]]:
    raise NotImplementedError


@functools.singledispatch
def _compile_initial_state(src) -> Callable[..., Tuple[torch.Tensor]]:
    raise NotImplementedError


@functools.singledispatch
def _compile_observables(src) -> Callable[..., Tuple[torch.Tensor]]:
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
def eval_observables(src, param_module: pyro.nn.PyroModule, X: State[T]) -> State[T]:
    raise NotImplementedError


@functools.singledispatch
def get_name(obj) -> str:
    raise NotImplementedError


@get_name.register
def _get_name_str(name: str) -> str:
    return name
