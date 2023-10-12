import os
import functools
import numbers
import operator
from typing import Callable, Dict, Optional, Tuple, Type, TypeVar, Union

import json
import mira
import mira.metamodel
import mira.metamodel.ops
import mira.modeling
import mira.modeling.petri
import mira.sources
import mira.sources.askenet
import mira.sources.askenet.petrinet
import mira.sources.askenet.regnet
import mira.sources.petri
import numpy
import pyro
import requests
import sympy
import sympytorch
import torch

from pyciemss.utils.distributions import mira_distribution_to_pyro

from chirho.dynamical.ops import InPlaceDynamics, State, simulate
from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.internals.backend import Solver

S = TypeVar("S")
T = TypeVar("T")

_InPlaceDynamicsMeta = type(InPlaceDynamics)
_PyroModuleMeta = type(pyro.nn.PyroModule)
class _CompiledInPlaceDynamicsMeta(_InPlaceDynamicsMeta, _PyroModuleMeta):
    pass

class CompiledInPlaceDynamics(pyro.nn.PyroModule, InPlaceDynamics, metaclass=_CompiledInPlaceDynamicsMeta):

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
    def diff(self, dX: State[torch.Tensor], X: State[torch.Tensor]) -> None:
        return eval_diff(self.src, self, dX, X)
    
    @pyro.nn.pyro_method
    def observation(self, X: State[torch.Tensor]) -> None:
        return eval_observation(self.src, self, X)
    
    def forward(self, 
                start_state: State[torch.Tensor], 
                start_time: torch.Tensor, 
                end_time: torch.Tensor, 
                solver: Solver=TorchDiffEq(),
                ):
        
        # Initialize random parameters once before simulating.
        # This is necessary because the parameters are PyroSample objects.
        for k in default_param_values(self.src).keys():
            getattr(self, get_name(k))

        return simulate(self, start_state, start_time, end_time, solver=solver)


    @functools.singledispatchmethod
    @classmethod
    def from_askenet(cls, src) -> "CompiledInPlaceDynamics":
        raise NotImplementedError

    @from_askenet.register
    @classmethod
    def _from_askenet_model(cls, src: mira.modeling.Model):
        model = cls(src)
        # Compile the numeric derivative of the model from the transition rate laws.
        setattr(model, "numeric_deriv", _compile_deriv(src))
        for trans in src.transitions.values():
            # These are not used in the compiled model, but are helpful for inspecting the rate laws.
            setattr(model, f"rate_law_{get_name(trans)}", _compile_rate_law(trans))

        # Compile the observation function from the model.
        for obs_var, obs in src.observables.items():
            setattr(model, f"observation_{get_name(obs_var)}", _compile_observable(obs))
        
        return model
    
    @from_askenet.register
    @classmethod
    def _from_askenet_path(cls, path: str):
        if "https://" in path:
            res = requests.get(path)
            model_json = res.json()
        else:
            if not os.path.exists(path):
                raise ValueError(f"Model file not found: {path}")
            with open(path) as fh:
                model_json = json.load(fh)
        return cls.from_askenet(model_json)

    @from_askenet.register
    @classmethod
    def _from_askenet_template_model(cls, template: mira.metamodel.TemplateModel):
        model = cls.from_askenet(mira.modeling.Model(template))

        # Check if all parameter names are strings
        if all(
            isinstance(param.key, str) for param in model.src.parameters.values()
        ):
            return model
        else:
            new_template = mira.metamodel.ops.aggregate_parameters(template)
            return cls.from_askenet(mira.modeling.Model(new_template))

    @from_askenet.register
    @classmethod
    def _from_askenet_json(cls, model_json: dict):
        # return a model from an ASKEM Model Representation json
        if "templates" in model_json:
            return cls.from_askenet(mira.metamodel.TemplateModel.from_json(model_json))
        elif "petrinet" in model_json["header"]["schema"]:
            return cls.from_askenet(
                mira.sources.askenet.petrinet.template_model_from_askenet_json(
                    model_json
                )
            )
        elif "regnet" in model_json["header"]["schema"]:
            return cls.from_askenet(
                mira.sources.askenet.regnet.template_model_from_askenet_json(model_json)
            )
        else:
            raise ValueError(f"Unknown model schema: {model_json['schema']}")


@functools.singledispatch
def eval_diff(src, param_module: pyro.nn.PyroModule, dX: State[T], X: State[T]):
    raise NotImplementedError


@functools.singledispatch
def eval_observation(src, param_module: pyro.nn.PyroModule, X: State[T]):
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
    numeric_deriv = sympytorch.SymPyModule(expressions=[simplified_deriv[get_name(var)] for var in src.variables.values()])
    return numeric_deriv

def _compile_rate_law(
    transition: mira.modeling.Transition,
) -> Callable[..., Tuple[torch.Tensor]]:
    return sympytorch.SymPyModule(
        expressions=[transition.template.rate_law.args[0]]
    )

def _compile_observable(
    observable: mira.modeling.Observable,
) -> Callable[..., Tuple[torch.Tensor]]:
    return sympytorch.SymPyModule(
        expressions=[observable.observable.expression.args[0]]
)

@eval_diff.register
def _eval_diff_compiled_mira(
    src: mira.modeling.Model,
    param_module: pyro.nn.PyroModule,
    dX: State[torch.Tensor],
    X: State[torch.Tensor],
):
    states = {s_name: getattr(X, s_name) for s_name in X.keys}

    parameters = {
        get_name(param_info): getattr(param_module, get_name(param_info))
        for param_info in src.parameters.values()
    }

    numeric_deriv = param_module.numeric_deriv(**states, **parameters)
    for i, var in enumerate(src.variables.values()):
        k = get_name(var)
        setattr(dX, k, numeric_deriv[i])


def normal_noise_model(name: str, obs_value: torch.Tensor) -> None:
    mean = obs_value
    var = 0.1 * obs_value
    return pyro.sample(name, pyro.distributions.Normal(mean, var))

@eval_observation.register
def _eval_observation_compiled_mira(
    src: mira.modeling.Model,
    param_module: pyro.nn.PyroModule,
    X: State[torch.Tensor],
    *,
    noise_kernel: Callable[[str, torch.Tensor], None] = normal_noise_model,
) -> None:
    
    states = {s_name: getattr(X, s_name) for s_name in X.keys}

    parameters = {
        get_name(param_info): getattr(param_module, get_name(param_info))
        for param_info in src.parameters.values()
    }

    for obs_var in src.observables.keys():
        obs_value = getattr(param_module, f"observation_{get_name(obs_var)}")(**states, **parameters)
        noise_kernel(get_name(obs_var), obs_value)
    

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


@default_param_values.register
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


@default_initial_state.register
def _mira_default_initial_state(src: mira.modeling.Model) -> State[torch.Tensor]:
    return State(
        **{
            get_name(var): torch.as_tensor(var.data["initial_value"])
            for var in src.variables.values()
        }
    )