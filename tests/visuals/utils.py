from typing import Dict, Optional, Iterable, Callable
import pandas as pd
import numpy as np
import torch

# Mildly related, I'm still using the 'datacube' format for testing...is there a better way (that loader leads to warnings, and I might be the only one using it now)


# originally from utils/inference_utils.py
def get_tspan(start, end, steps):
    """
    Thin wrapper around torch.linspace.
    """
    return torch.linspace(float(start), float(end), steps)


# originally from utils/interface_utils.py
def convert_to_output_format(
    samples: Dict[str, torch.Tensor],
    timepoints: Iterable[float],
    interventions: Optional[Dict[str, torch.Tensor]] = None,
    *,
    time_unit: Optional[str] = "(unknown)",
    quantiles: Optional[bool] = False,
    alpha_qs: Optional[Iterable[float]] = [
        0.01,
        0.025,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        0.975,
        0.99,
    ],
    num_quantiles: Optional[int] = 0,
    stacking_order: Optional[str] = "timepoints",
    observables: Optional[Dict[str, Callable]] = None,
    train_end_point: Optional[float] = None,
) -> pd.DataFrame:
    """
    Convert the samples from the Pyro model to a DataFrame in the TA4 requested format.

    time_unit -- Label timepoints in a semantically relevant way `timepoint_<time_unit>`.
       If None, a `timepoint_<time_unit>` field is not provided.
    """

    pyciemss_results = {"parameters": {}, "states": {}}

    for name, sample in samples.items():
        if sample.ndim == 1:
            # Any 1D array is a sample from the distribution over parameters.
            # Any 2D array is a sample from the distribution over states, unless it's a model weight.
            name = name + "_param"
            pyciemss_results["parameters"][name] = (
                sample.data.detach().cpu().numpy().astype(np.float64)
            )
        elif name == "model_weights":
            n_models = sample.shape[1]
            for i in range(n_models):
                pyciemss_results["parameters"][f"model_{i}_weight"] = (
                    sample[:, i].data.detach().cpu().numpy().astype(np.float64)
                )
        else:
            pyciemss_results["states"][name] = (
                sample.data.detach().cpu().numpy().astype(np.float64)
            )

    num_samples, num_timepoints = next(
        iter(pyciemss_results["states"].values())
    ).shape
    d = {
        "timepoint_id": np.tile(np.array(range(num_timepoints)), num_samples),
        "sample_id": np.repeat(np.array(range(num_samples)), num_timepoints),
    }

    # Parameters
    if interventions is None:
        d = {
            **d,
            **{
                k: np.repeat(v, num_timepoints)
                for k, v in pyciemss_results["parameters"].items()
            },
        }
    else:
        d = {
            **d,
            **assign_interventions_to_timepoints(
                interventions, timepoints, pyciemss_results["parameters"]
            ),
        }

    # Solution (state variables)
    d = {
        **d,
        **{
            k: np.squeeze(v.reshape((num_timepoints * num_samples, 1)))
            for k, v in pyciemss_results["states"].items()
        },
    }

    if observables is not None:
        expression_vars = {
            k.replace("_sol", ""): torch.squeeze(torch.tensor(d[k]), dim=-1)
            for k in pyciemss_results["states"].keys()
        }
        d = {
            **d,
            **{
                f"{observable_id}_obs": torch.squeeze(
                    expression(**expression_vars)
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                for observable_id, expression in observables.items()
            },
        }
    result = pd.DataFrame(d)
    if time_unit is not None:
        all_timepoints = result["timepoint_id"].map(lambda v: timepoints[v])
        result = result.assign(**{f"timepoint_{time_unit}": all_timepoints})

    if quantiles:
        result_q = make_quantiles(
            pyciemss_results,
            timepoints,
            alpha_qs,
            num_quantiles,
            time_unit=time_unit,
            stacking_order=stacking_order,
            train_end_point=train_end_point,
        )
        return result, result_q
    else:
        return result
