from typing import Dict, Any, List, Callable
from chirho.interventional.ops import Intervention
import torch


def param_value_objective(
    param_name: List[str],
    start_time: List[torch.Tensor],
    param_value: List[Intervention] = [None],
):

    def intervention_generator(
        x: torch.Tensor,
    ) -> Dict[float, Dict[str, Intervention]]:
        static_parameter_interventions = {}
        for count in range(len(param_name)):
            if param_value[count] is None:
                if not isinstance(param_value[count], Callable):
                    param_value[count] = lambda x: torch.tensor(x)
            static_parameter_interventions.update(
                {
                    start_time[count].item(): {
                        param_name[count]: param_value[count](x[count].item())
                    }
                }
            )
        return static_parameter_interventions

    return intervention_generator


def start_time_objective(param_name: List[str], param_value: List[Intervention]):

    def intervention_generator(
        x: torch.Tensor,
    ) -> Dict[float, Dict[str, Intervention]]:
        static_parameter_interventions = {}
        for count in range(len(param_name)):
            static_parameter_interventions.update(
                {x[count].item(): {param_name[count]: param_value[count]}}
            )
        return static_parameter_interventions

    return intervention_generator
