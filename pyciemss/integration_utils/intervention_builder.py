from typing import Callable, Dict, List

import torch
from chirho.interventional.ops import Intervention


def param_value_objective(
    param_name: List[str],
    start_time: List[torch.Tensor],
    param_value: List[Intervention] = [None],
) -> Callable[[torch.Tensor], Dict[float, Dict[str, Intervention]]]:
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
        assert x.size()[0] == param_size, (
            f"Size mismatch between input size ('{x.size()[0]}') and param_name size ('{param_size}'): "
            "check size for initial_guess_interventions and/or bounds_interventions."
        )
        static_parameter_interventions: Dict[float, Dict[str, Intervention]] = {}
        for count in range(param_size):
            if start_time[count].item() in static_parameter_interventions:
                static_parameter_interventions[start_time[count].item()].update(
                    {param_name[count]: param_value[count](x[count].item())}
                )
            else:
                static_parameter_interventions.update(
                    {
                        start_time[count].item(): {
                            param_name[count]: param_value[count](x[count].item())
                        }
                    }
                )
        return static_parameter_interventions

    return intervention_generator


def start_time_objective(
    param_name: List[str],
    param_value: List[Intervention],
) -> Callable[[torch.Tensor], Dict[float, Dict[str, Intervention]]]:
    param_size = len(param_name)

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
                    {param_name[count]: param_value[count]}
                )
            else:
                static_parameter_interventions.update(
                    {x[count].item(): {param_name[count]: param_value[count]}}
                )
        return static_parameter_interventions

    return intervention_generator


def start_time_param_value_objective(
    param_name: List[str],
    param_value: List[Intervention] = [None],
) -> Callable[[torch.Tensor], Dict[float, Dict[str, Intervention]]]:
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
                    {param_name[count]: param_value[count](x[count * 2 + 1].item())}
                )
            else:
                static_parameter_interventions.update(
                    {
                        x[count * 2].item(): {
                            param_name[count]: param_value[count](
                                x[count * 2 + 1].item()
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
    assert len(intervention_funcs) == len(intervention_func_lengths)

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
    interventions: List[Dict[float, Dict[str, Intervention]]]
) -> Dict[float, Dict[str, Intervention]]:
    static_parameter_interventions: Dict[float, Dict[str, Intervention]] = {}
    for intervention in interventions:
        for key, value in intervention.items():
            if key in static_parameter_interventions:
                static_parameter_interventions[key].update(value)
            else:
                static_parameter_interventions.update({key: value})
    return static_parameter_interventions
