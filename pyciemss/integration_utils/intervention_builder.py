from typing import Dict, Any, List
from chirho.interventional.ops import Intervention
import torch

def static_parameter_interventions_generator(
    x: torch.Tensor, static_parameter_interventions_def: Dict[str, List[Any]]
) -> Dict[float, Dict[str, Intervention]]:
    static_parameter_interventions = {}
    if static_parameter_interventions_def.get("start_time") is None:
        for count in range(len(static_parameter_interventions_def["param_name"])):
            static_parameter_interventions.update(
                {
                    x[count].item(): {
                        static_parameter_interventions_def["param_name"][
                            count
                        ]: static_parameter_interventions_def["param_value"][count]
                    }
                }
            )
    else:
        for count in range(len(static_parameter_interventions_def["param_name"])):
            static_parameter_interventions.update(
                {
                    static_parameter_interventions_def["start_time"][count].item(): {
                        static_parameter_interventions_def["param_name"][
                            count
                        ]: static_parameter_interventions_def["param_value"][count](
                            x[count].item()
                        )
                    }
                }
            )
    return static_parameter_interventions