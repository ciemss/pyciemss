import collections
from collections.abc import Callable
import functools
import json
import operator
import os
import warnings
from typing import Dict, List, Optional, Union, OrderedDict, Tuple
import requests
import networkx
import numpy
import torch
import pyro
import sympy
from sympytorch import SymPyModule

import mira
import mira.modeling
import mira.modeling.petri
import mira.metamodel
import mira.sources
import mira.sources.petri
import mira.sources.askenet.petrinet as petrinet
import mira.sources.askenet.regnet as regnet
from pyciemss.utils.distributions import ScaledBeta, mira_distribution_to_pyro
from pyro.distributions import Normal
from mira.metamodel.ops import aggregate_parameters

import bisect

from torchdiffeq import odeint

from pyciemss.interfaces import DynamicalSystem

from pyciemss.PetriNetODE.events import (Event, StaticEvent, StartEvent, ObservationEvent,
                                         LoggingEvent, StaticParameterInterventionEvent)

Time = Union[float, torch.tensor]
State = tuple[torch.tensor]
StateDeriv = tuple[torch.tensor]
Solution = Dict[str, torch.tensor]
class PetriNetODESystem(DynamicalSystem):
    '''
    Base class for ordinary differential equations models in PyCIEMSS.
    '''

    def __init__(self):
        super().__init__()
        # The order of the variables in the state vector used in the `deriv` method.
        self.var_order = self.create_var_order()
        self.total_population = None

    def reset(self) -> None:
        '''
        Resets the model to its initial state.
        '''
        self._observation_var_names = []
        self._static_events = []
        self._dynamic_stop_events = []
        self._observation_indices = {}
        self._observation_values = {}
        self._observation_indices_and_values_are_set_up = False

    def create_var_order(self) -> dict[str, int]:
        '''
        Returns the order of the variables in the state vector used in the `deriv` method.
        '''
        raise NotImplementedError

    def load_events(self, events: List[Event]) -> None:
        '''
        Loads a list of events into the model.
        '''
        for event in events:
            self.load_event(event)

    def load_event(self, event: Event) -> None:
        '''
        Loads an event into the model.
        '''
        # Execute specializations of the `_load_event` method, dispatched on the type of event.
        self._load_event(event)

        if isinstance(event, StaticEvent):
            # If the event is a start event, we need to assign
            # self.total_population to the sum of the initial populations.
            if isinstance(event, StartEvent):
                self.total_population = sum(event.initial_state.values())

            # If the event is a static event, then we need to set up the observation indices and values again.
            # We'll do this in the `forward` method if necessary.
            self._observation_indices_and_values_are_set_up = False
            bisect.insort(self._static_events, event)
        else:
            # If the event is a dynamic event, then we need to add it to the list of dynamic events.
            raise NotImplementedError

    @functools.singledispatchmethod
    def _load_event(self, event:Event) -> None:
        '''
        Loads an event into the model.
        '''
        pass

    @_load_event.register
    def _load_event_observation_event(self, event: ObservationEvent) -> None:
        # Add the variable names to the list of observation variable names if they are not already there.
        for var_name in event.observation.keys():
            if var_name not in self._observation_var_names:
                self._observation_var_names.append(var_name)

    def _setup_observation_indices_and_values(self):
        '''
        Set up the observation indices and observation values.
        '''
        if not self._observation_indices_and_values_are_set_up:
            self._observation_indices = {}
            self._observation_values = {}
            for var_name in self._observation_var_names:
                self._observation_indices[var_name] = [i for i, event in enumerate(self._static_events)
                                                       if isinstance(event, ObservationEvent)
                                                       and var_name in event.observation.keys()]
                self._observation_values[var_name] = torch.stack([self._static_events[i].observation[var_name]
                                                                  for i in self._observation_indices[var_name]])

            self._observation_indices_and_values_are_set_up = True

    def remove_start_event(self) -> None:
        '''
        Remove the start event from the model.
        '''
        self._remove_static_events(StartEvent)

    def remove_observation_events(self) -> None:
        '''
        Remove all observation events from the model.
        '''
        self._observation_var_names = []
        self._remove_static_events(ObservationEvent)

    def remove_logging_events(self) -> None:
        '''
        Remove all logging events from the model.
        '''
        self._remove_static_events(LoggingEvent)

    def remove_static_parameter_intervention_events(self) -> None:
        '''
        Remove all static parameter intervention events from the model.
        '''
        self._remove_static_events(StaticParameterInterventionEvent)

    def _remove_static_events(self, event_class) -> None:
        '''
        Remove all static events of Type `event_class` from the model.
        '''
        self._static_events = [event for event in self._static_events if not isinstance(event, event_class)]
        self._observation_indices_and_values_are_set_up = False

    def deriv(self, t: Time, state: State) -> StateDeriv:
        '''
        Returns a derivate of `state` with respect to `t`.
        '''
        raise NotImplementedError

    @pyro.nn.pyro_method
    def param_prior(self) -> None:
        '''
        Inplace method defining the prior distribution over model parameters.
        All random variables must be defined using `pyro.sample` or `PyroSample` methods.
        '''
        raise NotImplementedError

    @pyro.nn.pyro_method
    def observation_model(self, solution: Dict[str, torch.Tensor], var_name: str) -> None:
        '''
        Conditional distribution of observations given true state trajectory.
        All random variables must be defined using `pyro.sample` or `PyroSample` methods.
        This needs to be called once for each `var_name` in the set of observed variables.
        '''
        raise NotImplementedError

    def static_parameter_intervention(self, parameter: str, value: torch.Tensor):
        '''
        Inplace method defining how interventions are applied to modify the model parameters.
        '''
        raise NotImplementedError

    def setup_before_solve(self) -> None:
        '''
        Inplace method for setting up the model.
        '''
        self._setup_observation_indices_and_values()

    @pyro.nn.pyro_method
    def get_solution(self, method="dopri5") -> Solution:
        # Check that the start event is the first event
        assert isinstance(self._static_events[0], StartEvent), "Please initialize the model before sampling."

        # Load initial state
        initial_state = tuple(self._static_events[0].initial_state[v] for v in self.var_order.keys())

        # Get tspan from static events
        tspan = torch.tensor([e.time for e in self._static_events])

        solutions = [tuple(s.reshape(-1) for s in initial_state)]

        # Find the indices of the static intervention events
        bound_indices = [0] + [i for i, event in enumerate(self._static_events) if isinstance(event, StaticParameterInterventionEvent)] + [len(self._static_events)]
        bound_pairs = zip(bound_indices[:-1], bound_indices[1:])

        # Iterate through the static intervention events, running the ODE solver in between each.
        for (start, stop) in bound_pairs:

            if isinstance(self._static_events[start], StaticParameterInterventionEvent):
                # Apply the intervention
                self.static_parameter_intervention(self._static_events[start].parameter, self._static_events[start].value)

            # Construct a tspan between the current time and the next static intervention event
            local_tspan = tspan[start:stop+1]

            # Simulate from ODE with the new local tspan
            local_solution = odeint(self.deriv, initial_state, local_tspan, method=method)

            # Add the solution to the solutions list.
            solutions.append(tuple(s[1:] for s in local_solution))

            # update the initial_state
            initial_state = tuple(s[-1] for s in local_solution)

        # Concatenate the solutions
        solution = tuple(torch.cat(s) for s in zip(*solutions))
        solution = {v: solution[i] for i, v in enumerate(self.var_order.keys())}

        return solution

    @pyro.nn.pyro_method
    def add_observation_likelihoods(self, solution: Solution, observation_model=None) -> None:
        '''
        Compute likelihoods for observations.
        '''
        if observation_model is None:
            observation_model = self.observation_model

        for var_name in self._observation_var_names:
            observation_indices = self._observation_indices[var_name]
            observation_values = self._observation_values[var_name]
            filtered_solution = {v: solution[observation_indices] for v, solution in solution.items()}
            if hasattr(self, 'compile_observables_p') and self.compile_observables_p:
                for observable in self.compiled_observables:
                    filtered_solution[observable] = torch.squeeze(self.compiled_observables[observable](**filtered_solution), dim=-1)
            with pyro.condition(data={var_name: observation_values}):
                observation_model(filtered_solution, var_name)

    def log_solution(self, solution: Solution) -> Solution:
        '''
        This method wraps the solution in a pyro.deterministic call to ensure it is in the trace.
        '''
        # Log the solution
        logging_indices = [i for i, event in enumerate(self._static_events) if isinstance(event, LoggingEvent)]

        # Return the logged solution wrapped in a pyro.deterministic call to ensure it is in the trace
        logged_solution = {v: pyro.deterministic(f"{v}_sol", solution[logging_indices]) for v, solution in solution.items()}

        return logged_solution

## why is this here? It should be in MiraPetriNetODESystem if it is Mira specific
@functools.singledispatch
def get_name(obj) -> str:
    """
    Function to get a string-valued name for a MIRA object for use in a Pyro model.

    Guaranteed to be human-readable and unique for Variables,
    and merely unique for everything else.
    """
    raise NotImplementedError


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


class MiraPetriNetODESystem(PetriNetODESystem):
    """
    Create an ODE system from a petri-net specification.
    """
    def __init__(self, G: mira.modeling.Model, compile_rate_law_p=True, compile_observables_p=True, add_uncertainty=True):
        self.G = G
        self.compile_rate_law_p = compile_rate_law_p
        self.add_uncertainty = add_uncertainty
        if compile_rate_law_p:
            self.G.parameters = {
                param: value for param, value in self.G.parameters.items()
                if value and value.placeholder == False
            }
        super().__init__()
        self.compile_rate_law_p = compile_rate_law_p
        self.compile_observables_p = compile_observables_p
        if self.compile_rate_law_p:
            self.compiled_rate_law = self.compile_rate_law()
        if self.compile_observables_p:
            self.compiled_observables = self.compile_observables()
        if all(var.data.get("initial_value", None) is not None for var in self.var_order.values()):
            for var in self.var_order.values():
                self.register_buffer(
                    f"default_initial_state_{get_name(var)}",
                    torch.as_tensor(var.data["initial_value"])
                )

            self.default_initial_state = tuple(
                getattr(self, f"default_initial_state_{get_name(var)}", None)
                for var in self.var_order.values()
            )
        
        # Set up the parameters
        for param_info in G.parameters.values():

            param_distribution = param_info.distribution

            if param_distribution is not None:
                param_info.value = mira_distribution_to_pyro(param_distribution)
            else:
                param_value = param_info.value
                if param_value is None:
                    warnings_string = f"Parameter {get_name(param_info)} has value None and will be set to Uniform(0, 1). This is likely to be an error."
                    warnings.warn(warnings_string)
                    param_info.value = pyro.distributions.Uniform(0.0, 1.0)
                elif param_value <= 0:
                    warnings_string = f"Parameter {get_name(param_info)} has value {param_value} <= 0.0. This is likely to be an error."
                    warnings.warn(warnings_string)
                elif isinstance(param_value, (int, float)):
                    if self.add_uncertainty:
                        param_info.value = pyro.distributions.Uniform(max(0.9 * param_value, 0.0), 1.1 * param_value)
                else:
                    raise ValueError(f"Parameter {get_name(param_info)} has value {param_value} of type {type(param_value)} which is not supported.")

    def compile_observables(self) -> Dict[str, SymPyModule]:
        """Compile the observables during initialization."""

        # compute the symbolic observables
        return {
            observable_id: SymPyModule(expressions=[self.extract_sympy(symbolic_observable.observable.expression)])
            for observable_id, symbolic_observable in self.G.observables.items()
        }
        
    def compile_rate_law(self) -> Callable[[float, Tuple[torch.Tensor]], Tuple[torch.Tensor]]:
        """Compile the deriv function during initialization."""

        # compute the symbolic derivatives
        symbolic_derivs = {get_name(var): 0 for var in self.var_order.values()}
        for t in self.G.transitions.values():
            flux = self.extract_sympy(t.template.rate_law)
            for c in t.consumed:
                symbolic_derivs[get_name(c)] -= flux
            for p in t.produced:
                symbolic_derivs[get_name(p)] += flux

        # convert to a function
        numeric_derivs = SymPyModule(expressions=[symbolic_derivs[get_name(k)] for k in self.var_order.values()])
        return numeric_derivs
        
    def extract_sympy(self, sympy_expr_str: mira.metamodel.templates.SympyExprStr) -> sympy.Expr:
        """Convert the mira SympyExprStr to a sympy.Expr."""
        return sympy_expr_str.args[0]

    def create_var_order(self) -> dict[str, int]:
        '''
        Returns the order of the variables in the state vector used in the `deriv` method.
        Specialization of the base class method using the Mira graph object.
        '''
        return collections.OrderedDict(
            (get_name(var), var) for var in sorted(self.G.variables.values(), key=get_name)
        )
    
    @functools.singledispatchmethod
    @classmethod
    def from_mira(cls, model: mira.modeling.Model, **kwargs) -> "MiraPetriNetODESystem":
        return cls.from_askenet(model, **kwargs)

    @from_mira.register(mira.metamodel.TemplateModel)
    @classmethod
    def _from_template_model(cls, model_template: mira.metamodel.TemplateModel, **kwargs):
        return cls.from_askenet(model_template, **kwargs)

    @from_mira.register(dict)
    @classmethod
    def _from_json(cls, model_json: dict, **kwargs):
        return cls.from_mira(mira.metamodel.TemplateModel.from_json(model_json), **kwargs)

    @from_mira.register(str)
    @classmethod
    def _from_json_file(cls, model_json_path: str, **kwargs):
        if not os.path.exists(model_json_path):
            raise ValueError(f"Model file not found: {model_json_path}")
        with open(model_json_path, "r") as f:
            return cls.from_mira(json.load(f), **kwargs)

    @functools.singledispatchmethod
    @classmethod
    def from_askenet(cls, model: mira.modeling.Model, **kwargs) -> "MiraPetriNetODESystem":
        """Return a model from a MIRA model."""
        return cls(model, **kwargs)


    @from_askenet.register(mira.metamodel.TemplateModel)
    @classmethod
    def _from_template_model(cls, model_template: mira.metamodel.TemplateModel, **kwargs):
        """Return a model from a MIRA model template."""
        model = cls.from_askenet(mira.modeling.Model(model_template), **kwargs)

        # Check if all parameter names are strings
        # if all(isinstance(param.key, str) for param in model.G.parameters.values()):
        #     return model
        # else:
        #     new_template = aggregate_parameters(model_template)
        #     return cls.from_askenet(mira.modeling.Model(new_template), **kwargs)

        # Check if we wish to aggregate parameters (default is no)
        if 'aggregate_parameters_p' in kwargs and kwargs['aggregate_parameters_p']:
            new_template = aggregate_parameters(model_template)
            return cls.from_askenet(mira.modeling.Model(new_template), **kwargs)
        else:
            return cls.from_askenet(mira.modeling.Model(model_template), **kwargs)
            

    @from_askenet.register(dict)
    @classmethod
    def _from_json(cls, model_json: dict, **kwargs):
        """Return a model from an ASKEM Model Representation json."""
        if "templates" in model_json:
            return cls.from_askenet(mira.metamodel.TemplateModel.from_json(model_json), **kwargs)
        elif 'petrinet' in model_json['schema']:
            return cls.from_askenet(petrinet.template_model_from_askenet_json(model_json), **kwargs)
        elif 'regnet' in model_json['schema']:
            return cls.from_askenet(regnet.template_model_from_askenet_json(model_json), **kwargs)


    
    @from_askenet.register(str)
    @classmethod
    def _from_path(cls, model_json_path: str, **kwargs):
        """Return a model from an ASKEM Model Representation path (either url or local file)."""
        if "https://" in model_json_path:
            res = requests.get(model_json_path)
            model_json = res.json()
        else:
            if not os.path.exists(model_json_path):
                raise ValueError(f"Model file not found: {model_json_path}")
            with open(model_json_path) as fh:
                model_json = json.load(fh)
        return cls.from_askenet(model_json, **kwargs)

    # Note: This code below referred to a class that doesn't exist (or at least isn't imported).

    # def to_askenet_petrinet(self) -> dict:
    #     """Return an ASKEM Petrinet Model Representation json."""
    #     return AskeNetPetriNetModel(self.G).to_json()

    # def to_askenet_regnet(self) -> dict:
    #     """Return an ASKEM Regnet Model Representation json."""
    #     return AskeNetRegNetModel(self.G).to_json()
    
    def to_networkx(self) -> networkx.MultiDiGraph:
        from pyciemss.utils.petri_utils import load
        return load(mira.modeling.petri.PetriNetModel(self.G).to_json())

    @pyro.nn.pyro_method
    def deriv(self, t: Time, state: State) -> StateDeriv:
        if self.compile_rate_law_p:
            # Get the current state
            states = {v: state[i] for i, v in enumerate(self.var_order.keys())}
            # Get the parameters
            parameters = {get_name(param_info): getattr (self, get_name(param_info))
                          for param_info in self.G.parameters.values()
                        }
            
            # Evaluate the rate laws for each transition
            deriv_tensor = self.compiled_rate_law(**states, **parameters, **dict(t=t))
            return tuple(deriv_tensor[i] for i in range(deriv_tensor.shape[0]))
        else:
            return self.mass_action_deriv(t, state)


    
    def mass_action_deriv(self, t: Time, state: State) -> StateDeriv:
        states = {k: state[i] for i, k in enumerate(self.var_order.values())}
        derivs = {k: 0. for k in states}

        population_size = sum(states.values())

        for transition in self.G.transitions.values():
            rate_law = transition.template.rate_law
            
            flux = getattr(self, get_name(transition.rate)) * functools.reduce(
                operator.mul, [states[k] for k in transition.consumed], 1
            )
            if len(transition.control) > 0:
                flux = flux * functools.reduce(operator.add, [states[k] for k in transition.control]) / population_size**len(transition.control)

            for c in transition.consumed:
                derivs[c] -= flux
            for p in transition.produced:
                derivs[p] += flux

        return tuple(derivs[v] for v in self.var_order.values())

    
    @pyro.nn.pyro_method
    def param_prior(self):
        for param_info in self.G.parameters.values():
            param_name = get_name(param_info)
            param_value = param_info.value
            if isinstance(param_value, torch.nn.Parameter):
                setattr(self, param_name, pyro.param(param_name, param_value))
            elif isinstance(param_value, pyro.distributions.Distribution):
                setattr(self, param_name, pyro.sample(param_name, param_value))
            elif isinstance(param_value, (int, float, numpy.ndarray, torch.Tensor)):
                setattr(self, param_name, pyro.deterministic(param_name, torch.as_tensor(param_value)))
            else:
                raise TypeError(f"Unknown parameter type: {type(param_value)}")

    @pyro.nn.pyro_method
    def observation_model(self, solution: Solution, var_name: str) -> None:
        # Default implementation just records the observation, with no randomness.
        pyro.deterministic(var_name, solution[var_name])

    def static_parameter_intervention(self, parameter: str, value: torch.Tensor):
        setattr(self, get_name(self.G.parameters[parameter]), value)

    def __repr__(self):
        par_string = ",\n\t".join([f"{get_name(p)} = {p.value}" for p in self.G.parameters.values()])
        return f"{self.__class__.__name__}(\n\t{par_string})"

class ScaledNormalNoisePetriNetODESystem(MiraPetriNetODESystem):
    '''
    This is a wrapper around PetriNetODESystem that adds Gaussian noise to the ODE system.
    '''
    def __init__(self, G: mira.modeling.Model, noise_scale: float = 0.1, compile_rate_law_p: bool = False, compile_observables_p: bool = False,  **kwargs):
        super().__init__(G, compile_rate_law_p=compile_rate_law_p, compile_observables_p=compile_observables_p, **kwargs)
        self.register_buffer("noise_scale", torch.as_tensor(noise_scale))
        assert self.noise_scale > 0, "Noise scale must be positive"
        assert self.noise_scale <= 1, "Noise scale must be less than 1"

    def __repr__(self):
        par_string = ",\n\t".join([f"{get_name(p)} = {p.value}" for p in self.G.parameters.values()])
        noise_string = f"noise_scale = {self.noise_scale}"
        return f"{self.__class__.__name__}(\n\t{par_string},\n\t{noise_string}\n)"
    
    @pyro.nn.pyro_method
    def observation_model(self, solution: Solution, var_name: str) -> None:
        mean = solution[var_name]
        # Scale the std dev by the mean, with some minimum
        scale = self.noise_scale * torch.maximum(mean, torch.as_tensor(0.005 * self.total_population))
        pyro.sample(var_name, Normal(mean, scale).to_event(1))

        
class ScaledBetaNoisePetriNetODESystem(MiraPetriNetODESystem):
    '''
    This is a wrapper around PetriNetODESystem that adds Beta noise to the ODE system.
    '''
    def __init__(self, G: mira.modeling.Model, pseudocount: float = 1., *, noise_scale=None, compile_rate_law_p: bool=False, compile_observables_p: bool=False, **kwargs):
        super().__init__(G, compile_rate_law_p=compile_rate_law_p, compile_observables_p=compile_observables_p, **kwargs)
        self.parameterized_by_pseudocount = noise_scale is None
        if self.parameterized_by_pseudocount:
            self.register_buffer("pseudocount", torch.as_tensor(pseudocount))
        else:
            self.register_buffer("noise_scale", torch.as_tensor(noise_scale))

    def __repr__(self):
        par_string = ",\n\t".join([f"{get_name(p)} = {p.value}" for p in self.G.parameters.values()])
        count_string = f"pseudocount = {self.pseudocount}"
        return f"{self.__class__.__name__}(\n\t{par_string},\n\t{count_string}\n)"

    @pyro.nn.pyro_method
    def observation_model(self, solution: Solution, var_name: str) -> None:
        mean = torch.maximum(solution[var_name], torch.tensor(1e-9))
        if self.parameterized_by_pseudocount:
            pyro.sample(var_name, ScaledBeta(mean, self.total_population, pseudocount=self.pseudocount).to_event(1))
        else:
            scale = self.noise_scale * torch.maximum(mean, torch.as_tensor(0.005 * self.total_population))
            pyro.sample(var_name, ScaledBeta(mean, self.total_population, scale=scale).to_event(1))
