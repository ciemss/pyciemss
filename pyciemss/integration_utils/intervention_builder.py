from typing import Callable, Dict, List

import torch
from chirho.interventional.ops import Intervention


def param_value_objective(
    param_name: List[str],
    start_time: List[torch.Tensor],
    param_value: List[Intervention] = [None],
) -> Callable[[torch.Tensor], Dict[float, Dict[str, Intervention]]]:
    """
    Create static parameter intervention template for optimizing parameter values.

    Args:
        param_name (List[str]): A list of parameter names to associate with interventions.
        start_time (List[torch.Tensor]): A list of start times for the interventions.
        param_value (List[Intervention]): A list of parameter values (or callable functions
            that return the values). Default is a list of None values, in which case a simple identity
            function is used.

    Returns:
        intervention_generator (Callable[[torch.Tensor], Dict[float, Dict[str, Intervention]]]): A function that,
        when given an input tensor `x` for parameter values, returns a dictionary of interventions at teh specified
        times.
    """
    param_size = len(param_name)
    if len(param_value) < param_size and param_value[0] is None:
        param_value = [None for _ in param_name]
    for count in range(param_size):
        if param_value[count] is None:
            if not callable(param_value[count]):
                # Note that param_value needs to be Callable
                param_value[count] = lambda y: torch.tensor(y)

    def intervention_generator(
        x: torch.Tensor,
    ) -> Dict[float, Dict[str, Intervention]]:
        x = torch.atleast_1d(x)
        assert x.size()[0] == param_size, (
            f"Size mismatch between input size ('{x.size()[0]}') and param_name size ('{param_size}'): "
            "check size for initial_guess_interventions and/or bounds_interventions."
        )
        static_parameter_interventions: Dict[float, Dict[str, Intervention]] = {}
        for count in range(param_size):
            if start_time[count].item() in static_parameter_interventions:
                static_parameter_interventions[start_time[count].item()].update(
                    {
                        param_name[count]: torch.atleast_1d(
                            param_value[count](x[count].item())
                        )
                    }
                )
            else:
                static_parameter_interventions.update(
                    {
                        start_time[count].item(): {
                            param_name[count]: torch.atleast_1d(
                                param_value[count](x[count].item())
                            )
                        }
                    }
                )
        return static_parameter_interventions

    return intervention_generator


def start_time_objective(
    param_name: List[str],
    param_value: List[torch.Tensor],
) -> Callable[[torch.Tensor], Dict[float, Dict[str, Intervention]]]:
    """
    Create static parameter intervention template for optimizing start times.

    Args:
        param_name (List[str]): A list of parameter names to associate with interventions.
        param_value (List[torch.Tensor]): A list of parameter values (as tensors) corresponding to the parameters.

    Returns:
        intervention_generator (Callable[[torch.Tensor], Dict[float, Dict[str, Intervention]]]): A function that,
        when given an input tensor `x` for start times, returns a dictionary of interventions at those times.
    """
    param_size = len(param_name)
    # Note: code below will only work for tensors and not callable functions
    param_value = [torch.atleast_1d(y) for y in param_value]

    def intervention_generator(
        x: torch.Tensor,
    ) -> Dict[float, Dict[str, Intervention]]:
        x = torch.atleast_1d(x)
        assert x.size()[0] == param_size, (
            f"Size mismatch between input size ('{x.size()[0]}') and param_name size ('{param_size}'): "
            "check size for initial_guess_interventions and/or bounds_interventions."
        )
        static_parameter_interventions: Dict[float, Dict[str, Intervention]] = {}
        for count in range(param_size):
            if x[count].item() in static_parameter_interventions:
                static_parameter_interventions[x[count].item()].update(
                    {param_name[count]: torch.atleast_1d(param_value[count])}
                )
            else:
                static_parameter_interventions.update(
                    {
                        x[count].item(): {
                            param_name[count]: torch.atleast_1d(param_value[count])
                        }
                    }
                )
        return static_parameter_interventions

    return intervention_generator


def start_time_param_value_objective(
    param_name: List[str],
    param_value: List[Intervention] = [None],
) -> Callable[[torch.Tensor], Dict[float, Dict[str, Intervention]]]:
    """
    Create static parameter intervention template for optimizing start times and parameter values at the same time.

    Args:
        param_name (List[str]): A list of parameter names to associate with interventions.
        param_value (List[Intervention]): A list of parameter values (or callable functions
            that return the values). Default is a list of None values, in which case a simple identity
            function is used.

    Returns:
        intervention_generator (Callable[[torch.Tensor], Dict[float, Dict[str, Intervention]]]): A function that,
        when given an input tensor `x`, returns a dictionary of interventions at specific times.
    """
    param_size = len(param_name)
    if len(param_value) < param_size and param_value[0] is None:
        param_value = [None for _ in param_name]
    for count in range(param_size):
        if param_value[count] is None:
            if not callable(param_value[count]):
                param_value[count] = lambda y: torch.tensor(y)

    def intervention_generator(
        x: torch.Tensor,
    ) -> Dict[float, Dict[str, Intervention]]:
        x = torch.atleast_1d(x)
        assert x.size()[0] == param_size * 2, (
            f"Size mismatch between input size ('{x.size()[0]}') and param_name size ('{param_size * 2}'): "
            "check size for initial_guess_interventions and/or bounds_interventions."
        )
        static_parameter_interventions: Dict[float, Dict[str, Intervention]] = {}
        for count in range(param_size):
            if x[count * 2].item() in static_parameter_interventions:
                static_parameter_interventions[x[count * 2].item()].update(
                    {
                        param_name[count]: torch.atleast_1d(
                            param_value[count](x[count * 2 + 1].item())
                        )
                    }
                )
            else:
                static_parameter_interventions.update(
                    {
                        x[count * 2].item(): {
                            param_name[count]: torch.atleast_1d(
                                param_value[count](x[count * 2 + 1].item())
                            )
                        }
                    }
                )
        return static_parameter_interventions

    return intervention_generator


def intervention_func_combinator(
    intervention_funcs: List[
        Callable[[torch.Tensor], Dict[float, Dict[str, Intervention]]]
    ],
    intervention_func_lengths: List[int],
) -> Callable[[torch.Tensor], Dict[float, Dict[str, Intervention]]]:
    """
    Combine multiple intervention functions into a single function.

    Args:
        intervention_funcs (List[Callable[[torch.Tensor], Dict[float, Dict[str, Intervention]]]]):
            A list of intervention functions to be combined.
        intervention_func_lengths (List[int]): A list of lengths corresponding to the input size expected
            by each intervention function.

    Returns:
        intervention_generator (Callable[[torch.Tensor], Dict[float, Dict[str, Intervention]]]): A combined
        intervention function that calls each individual function with the appropriate portion of the input tensor.
    """
    assert len(intervention_funcs) == len(intervention_func_lengths), (
        f"Size mismatch between number of intervention functions ('{len(intervention_funcs)}')"
        f"and number of intervention function lengths ('{len(intervention_func_lengths)}') "
    )

    total_length = sum(intervention_func_lengths)

    # Note: This only works for combining static parameter interventions.
    def intervention_generator(
        x: torch.Tensor,
    ) -> Dict[float, Dict[str, Intervention]]:
        x = torch.atleast_1d(x)
        assert x.size()[0] == total_length
        interventions: List[Dict[float, Dict[str, Intervention]]] = [
            {} for _ in range(len(intervention_funcs))
        ]
        i = 0
        for j, (input_length, intervention_func) in enumerate(
            zip(intervention_func_lengths, intervention_funcs)
        ):
            interventions[j] = intervention_func(x[i : i + input_length])
            i += input_length
        return combine_static_parameter_interventions(interventions)

    return intervention_generator


def combine_static_parameter_interventions(
    interventions: List[Dict[float, Dict[str, Intervention]]],
) -> Dict[float, Dict[str, Intervention]]:
    """
    Combine a list of static parameter interventions into one.

    Args:
        interventions (List[Dict[float, Dict[str, Intervention]]]): A list of static parameter interventions to be
        combined.

    Returns:
        static_parameter_interventions (Dict[float, Dict[str, Intervention]]): A single dictionary containing all
        combined interventions.
    """
    static_parameter_interventions: Dict[float, Dict[str, Intervention]] = {}
    for intervention in interventions:
        for key, value in intervention.items():
            if key in static_parameter_interventions:
                static_parameter_interventions[key].update(value)
            else:
                static_parameter_interventions.update({key: value})
    return static_parameter_interventions
