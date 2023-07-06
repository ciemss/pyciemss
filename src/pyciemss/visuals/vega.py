from typing import List, Dict, Any, Callable, Union
from numbers import Integral, Number

from copy import deepcopy
import IPython.display
import pandas as pd
import numpy as np
import pkgutil
import json
import re

VegaSchema = Dict[str, Any]


def _histogram_multi_schema() -> VegaSchema:
    return json.loads(pkgutil.get_data(__name__, "histogram_static_bins_multi.vg.json"))


def _trajectory_schema() -> VegaSchema:
    return json.loads(pkgutil.get_data(__name__, "trajectories.vg.json"))


# Trajectory Visualizations ------------------
def trajectories(
    distributions: Union[None, pd.DataFrame] = None,
    *,
    traces: Union[None, pd.DataFrame] = None,
    points: Union[None, pd.DataFrame] = None,
    subset: Union[str, list] = all,
    qlow: float = 0.05,
    qhigh: float = 0.95,
    limit: Union[None, Integral] = None,
    colors: Union[None, dict] = None,
    relabel: Union[None, Dict[str, str]] = None,
) -> VegaSchema:
    """_summary_

    TODO: Interpolation method probably needs attention...
    TODO: Intervention marker line

    Args:
        observations (_type_): Dataframe formatted per pyciemss.utils.interface_utils.convert_to_output_format
           These will be plotted as either spans (multiples sample_ids present) or lines (only one sample_id).
        points (_type_): Example points to plot (joined by lines)
        subset (any, optional): Subset the 'observations' based on keys/values.
           - Default is the 'all' function, and it keeps all keys
           - If a string is present, it is treated as a regex and matched against the key
           - Otherwise, assumed tob e a list-like of keys to keep
           If subset is specified, the color scale ordering follows the subset order.
        colors: Use the specified colors as a pre-relable keyed dictionary to vega-valid color.
           Mapping to None will drop that sequence (can be used in addition to or instead of subset)
        relabel (None, Dict[str, str]): Relabel elements for rendering.  Happens
            after key subsetting.
        limit --
    """
    if relabel is None:
        relabel = dict()

    distributions = distributions.set_index(["timepoint_id", "sample_id"])

    if subset == all:
        keep = distributions.columns
    elif isinstance(subset, str):
        keep = [k for k in distributions.columns if re.match(subset, k)]
    else:
        keep = subset

    if colors:
        keep = [k for k in keep if colors.get(k, None) is not None]

    distributions = distributions.filter(items=keep).rename(columns=relabel)

    def _quantiles(g):
        return pd.Series(
            {
                "lower": g.quantile(q=qlow).values[0],
                "upper": g.quantile(q=qhigh).values[0],
            }
        )

    if distributions is not None:
        distributions = (
            distributions.melt(ignore_index=False, var_name="trajectory")
            .set_index("trajectory", append=True)
            .groupby(level=["trajectory", "timepoint_id"])
            .apply(_quantiles)
            .reset_index()
            .iloc[:limit]
            .to_dict(orient="records")
        )
    else:
        distributions = []

    if traces is not None:
        traces = (
            traces.melt(ignore_index=False, var_name="trajectory")
            .reset_index()
            .to_dict(orient="records")
        )
    else:
        traces = []

    if points is not None:
        points = (
            points.set_index(["timepoint_id", "sample_id"])
            .melt(ignore_index=False, var_name="trajectory")
            .reset_index()
            .to_dict(orient="records")
        )
    else:
        points = []

    schema = _trajectory_schema()
    schema["data"] = replace_named_with(
        schema["data"], "distributions", ["values"], distributions
    )

    schema["data"] = replace_named_with(schema["data"], "points", ["values"], points)
    schema["data"] = replace_named_with(schema["data"], "traces", ["values"], traces)

    if colors is not None:
        colors = {relabel.get(k, k): v for k, v in colors.items()}
        schema["scales"] = replace_named_with(
            schema["scales"], "color", ["domain"], [*colors.keys()]
        )
        schema["scales"] = replace_named_with(
            schema["scales"], "color", ["range"], [*colors.values()]
        )

    return schema


# Things to check:
# _trajectories(prior_samples, tspan) == [*prior_samples.keys()]
# _trajectories(prior_samples, tspan, obs_keys = all) == [*prior_samples.keys()]
# _trajectories(prior_samples, tspan, obs_keys = ".*_sol") == ['Rabbits_sol', 'Wolves_sol']
# combinations of calls (colors, colors+relable, colors+subset, relable+subset, relabel+colors+subset, etc)

# Called like:
# plot = vega.trajectories(prior_samples, tspan, obs_keys=".*_sol")
# with open("trajectories.json", "w") as f:
#     json.dump(plot, f, indent=3)
# vega.ipy_display(plot)


# Histogram visualizations ------------------
def sturges_bin(data):
    """Determine number of bin susing sturge's rule.
    TODO: Consider Freedman-Diaconis (larger data sizes and spreads)
    """
    return int(np.ceil(np.log2(len(data))) + 1)


def histogram_multi(
    *,
    xrefs: List[Number] = [],
    yrefs: List[Number] = [],
    bin_rule: Callable = sturges_bin,
    return_bins: bool = False,
    **data,
) -> VegaSchema:
    """
    Create a histogram with server-side binning.

    TODO: Maybe compute overlap in bins explicitly and visualize as stacked?
          Limits (practically) to two distributions, but legend is more clear.

    TODO: Need to align histogram bin size between groups to make the visual
          representation more interpretable

    **data -- Datasets, name will be used to build labels.
    bin_rule -- Determines bins width using this function.
                Will received a joint dataframe of all data passed
    xrefs - List of values in the bin-range to highlight as vertical lines
    yrefs - List of values in the count-range to highlight as horizontal lines
    bins - Number of bins to divide into
    """

    schema = _histogram_multi_schema()

    def hist(label, subset, edges):
        assignments = np.digitize(subset, edges) - 1
        counts = np.bincount(assignments)
        spans = [*(zip(edges, edges[1:]))]
        desc = [
            {"bin0": l.item(), "bin1": h.item(), "count": c.item(), "label": label}
            for ((l, h), c) in zip(spans, counts)
        ]
        return desc

    def as_value_list(label, data):
        try:
            return data["state_values"].rename(label)
        except BaseException:
            return pd.Series(data).rename(label)

    data = {k: as_value_list(k, subset) for k, subset in data.items()}

    joint = pd.concat(data)
    bins_count = bin_rule(joint)
    _, edges = np.histogram(joint, bins=bins_count)

    hists = {k: hist(k, subset, edges) for k, subset in data.items()}
    desc = [item for sublist in hists.values() for item in sublist]

    schema["data"] = replace_named_with(schema["data"], "binned", ["values"], desc)

    schema["data"] = replace_named_with(
        schema["data"], "xref", ["values"], [{"value": v} for v in xrefs]
    )

    schema["data"] = replace_named_with(
        schema["data"], "yref", ["values"], [{"count": v} for v in yrefs]
    )

    if return_bins:
        return schema, pd.DataFrame(desc).set_index(["bin0", "bin1"])
    else:
        return schema


# --- Utlity functions for working with Vega ----
def ipy_display(spec: Dict[str, Any], *, lite=False):
    """Wrap for dispaly in an ipython notebook.
    spec -- A vega JSON schema ready for rendering
    """
    if lite:
        bundle = {"application/vnd.vegalite.v5+json": spec}
    else:
        bundle = {"application/vnd.vega.v5+json": spec}

    return IPython.display.display(bundle, raw=True)


def save_schema(schema: Dict[str, Any], path: str):
    with open(path, "w") as f:
        json.dump(schema, f, indent=3)


def resize(schema: VegaSchema, *, w: int = None, h: int = None) -> VegaSchema:
    """Utility for changing the size of a schema.
    Always returns a copy of the original schema.

    schema -- Schema to change
    w -- New width (optional)
    h -- New height (optional)
    """
    schema = deepcopy(schema)

    if h is not None:
        schema["height"] = h
    if w is not None:
        schema["width"] = w

    return schema


def pad(schema: VegaSchema, qty: Union[None, Number] = None) -> VegaSchema:
    """Add padding to a schema.

    Args:
        schema (VegaSchema): Schema to update
        qty (None | Number): Value to set padding to
         If None, removes padding if present

    Returns:
        VegaSchema: Schema with modified padding
    """
    schema = deepcopy(schema)
    if qty is None and "padding" in schema:
        del schema["padding"]
    else:
        schema["padding"] = qty

    return schema


def replace_named_with(ls: List, name: str, path: List[Any], new_value: Any) -> List:
    """Rebuilds the element with the given 'name' entry.

    An element is "named" if it has a key named "name".
    Only replaces the FIRST item so named.

    name -- Name to look for
    path -- Steps to the thing to actually replace
    new_value -- Value to place at end of path
    Path is a list of steps to take into the named entry.

    """

    def _maybe_replace(e, done_replacing):
        if done_replacing[0]:
            return e

        if "name" in e and e["name"] == name:
            e = deepcopy(e)
            part = e
            for step in path[:-1]:
                part = part[step]
            part[path[-1]] = new_value
            done_replacing[0] = True

        return e

    done_replacing = [False]
    updated = [_maybe_replace(e, done_replacing) for e in ls]

    if not done_replacing[0]:
        raise ValueError(f"Attempted to replace, but {name=} not found.")

    return updated
