from __future__ import annotations

import functools
from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Union

import mira
import mira.metamodel
import mira.modeling
import mira.sources
import mira.sources.amr
import pyro
import torch
from chirho.dynamical.handlers import LogTrajectory
from chirho.dynamical.internals._utils import _squeeze_time_dim
from chirho.dynamical.ops import State, simulate

S = TypeVar("S")
T = TypeVar("T")


class CompiledDynamics(pyro.nn.PyroModule):
    def __init__(self, src, **kwargs):
        super().__init__()
        self.src = src

        try:
            params = _compile_param_values(self.src)
        except Exception as e:
            raise ValueError(
                "The model parameters could not be compiled. Please check the model definition."
            ) from e

        for k, v in params.items():
            if hasattr(self, get_name(k)):
                continue

            if isinstance(
                v, (torch.nn.Parameter, pyro.nn.PyroParam, pyro.nn.PyroSample)
            ):
                setattr(self, f"persistent_{get_name(k)}", v)
            elif isinstance(v, torch.Tensor):
                self.register_buffer(f"persistent_{get_name(k)}", v)

        # Compile the numeric derivative of the model from the transition rate laws.
        try:
            setattr(self, "numeric_deriv_func", _compile_deriv(src))
        except Exception as e:
            raise ValueError(
                "The model derivative could not be compiled. Please check the model definition."
            ) from e

        try:
            setattr(self, "numeric_initial_state_func", _compile_initial_state(src))
        except Exception as e:
            raise ValueError(
                "The model initial state could not be compiled. Please check the model definition."
            ) from e

        try:
            setattr(self, "numeric_observables_func", _compile_observables(src))
        except Exception as e:
            raise ValueError(
                "The model observables could not be compiled. Please check the model definition."
            ) from e

        self.instantiate_parameters()

    @pyro.nn.pyro_method
    def deriv(self, X: State[torch.Tensor]) -> None:
        try:
            return eval_deriv(self.src, self, X)
        except Exception as e:
            raise ValueError(
                "The model derivative could not be evaluated. Please check the model definition. "
                "This could be due to to a missing state variable or parameter, "
                "or an error in the derivative definition."
            ) from e

    @pyro.nn.pyro_method
    def initial_state(self) -> State[torch.Tensor]:
        try:
            return eval_initial_state(self.src, self)
        except Exception as e:
            raise ValueError(
                "The model initial state could not be evaluated. Please check the model definition. "
                "This could be due to to a missing state variable or parameter, "
                "or an error in the initial state definition."
            ) from e

    @pyro.nn.pyro_method
    def observables(self, X: State[torch.Tensor]) -> State[torch.Tensor]:
        try:
            return eval_observables(self.src, self, X)
        except Exception as e:
            raise ValueError(
                "The model observables could not be evaluated. Please check the model definition. "
                "This could be due to to a missing state variable or parameter, "
                "or an error in the observables definition."
            ) from e

    @pyro.nn.pyro_method
    def instantiate_parameters(self):
        # Initialize random parameters once before simulating.
        # This is necessary because the parameters are PyroSample objects.
        for k, param in _compile_param_values(self.src).items():
            param_name = get_name(k)
            # Separating the persistent parameters from the non-persistent ones
            # is necessary because the persistent parameters are PyroSample objects representing the distribution,
            # and should not be modified during intervention.
            param_val = getattr(self, f"persistent_{param_name}")
            if isinstance(param, torch.Tensor):
                pyro.deterministic(f"persistent_{param_name}", param_val)
            self.register_buffer(param_name, param_val)

    def forward(
        self,
        start_time: torch.Tensor,
        end_time: torch.Tensor,
        logging_times: Optional[torch.Tensor] = None,
        is_traced: bool = False,
    ):
        self.instantiate_parameters()

        if logging_times is not None:
            with LogObservables(logging_times, self) as lo:
                try:
                    simulate(self.deriv, self.initial_state(), start_time, end_time)
                except AssertionError as e:
                    if str(e) == "AssertionError: underflow in dt nan":
                        raise AssertionError(
                            "Underflow in the adaptive time step size. "
                            "This is likely due to a stiff system of ODEs. "
                            "Try changing the (distribution on) parameters, the rate laws, "
                            "the initial state, or the time span."
                        ) from e
                    else:
                        raise e

                state = lo.state
                observables = lo.observables
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
        return cls.load(mira.modeling.Model(template))

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


class LogObservables(pyro.poutine.messenger.Messenger):
    def __init__(self, times: torch.Tensor, model: CompiledDynamics):
        super().__init__()
        self.model = model
        self.observables: State[torch.Tensor] = {}
        self.state: State[torch.Tensor] = {}

        # This gets around the issue of the LogTrajectory handler blocking `self`
        self.lt = LogTrajectory(times)

        self.reset()

    def reset(self):
        self._observables_names: List[str] = []
        self._initial_observables: State[torch.Tensor] = {}

    def _pyro_simulate_point(self, msg):
        self.lt._pyro_simulate_point(msg)

    def _pyro_post_simulate_trajectory(self, msg):
        observables = self.model.observables(msg["value"])
        self._observables_names = list(observables.keys())
        msg["value"] = {**msg["value"], **observables}

    def _pyro_simulate(self, msg):
        self._initial_observables = _squeeze_time_dim(
            self.model.observables(msg["args"][1])
        )

    def _pyro_post_simulate(self, msg):
        msg["args"] = (
            msg["args"][0],
            {**msg["args"][1], **self._initial_observables},
            msg["args"][2],
            msg["args"][3],
        )

        self.lt._pyro_post_simulate(msg)

        self.observables = {
            k: v for k, v in self.lt.trajectory.items() if k in self._observables_names
        }
        self.state = {
            k: v
            for k, v in self.lt.trajectory.items()
            if k not in self._observables_names
        }

        self.reset()
