from typing import Any, Dict


def check_solver(solver_method: str, solver_options: Dict[str, Any]) -> None:
    """
    Check if the solver method and options are valid.

    Args:
        solver_method (str): The name of the solver method.
        solver_options (dict): The options for the solver method.

    Raises:
        ValueError: If the solver method or options are invalid.
    """
    if solver_method == "euler":
        if "step_size" not in solver_options:
            raise ValueError(
                "The 'step_size' option is required for the 'euler' solver method."
                "Please provide a value for 'step_size' in the 'solver_options' dictionary."
            )
