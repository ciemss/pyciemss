import os
from typing import List, Dict, Any, Callable, Union
from collections.abc import Iterable
from numbers import Integral, Number

from copy import deepcopy
import IPython.display
import pandas as pd
import numpy as np
import torch
import pkgutil
import json
import re

from itertools import tee, filterfalse

VegaSchema = Dict[str, Any]


def _histogram_multi_schema() -> VegaSchema:
    return json.loads(pkgutil.get_data(__name__, "histogram_static_bins_multi.vg.json"))


def _trajectory_schema() -> VegaSchema:
    return json.loads(pkgutil.get_data(__name__, "trajectories.vg.json"))


# General Utilities ---------------
def partition(
    pred: Callable[[Any], bool], iterable: Iterable[Any]
) -> tuple[List[Any], List[Any]]:
    t1, t2 = tee(iterable)
    return filterfalse(pred, t1), filter(pred, t2)


def tensor_dump(tensors: Any, path: os.PathLike) -> None:
    reformatted = {k: v.detach().numpy().tolist() for k, v in tensors.items()}

    with open(path, "w") as f:
        json.dump(reformatted, f)


def tensor_load(path: os.PathLike) -> Dict[Any, Any]:
    with open(path) as f:
        data = json.load(f)

    data = {k: torch.from_numpy(np.array(v)) for k, v in data.items()}

    return data


# Trajectory Visualizations ------------------
def temporalize_datapoints(
    data: Union[list, Dict[str, Any]], timepoints: list, *, label=None
):
    """Assign timepoints to data. Will filter the observations
    to only those shape-compatible with the timepoints.
    The returned data is formatted for the 'observations' or 'points'
    arguments of the 'trajectories' plot.

    Args:
        data (Union[list, Dict[str, Any]]): Observations dictionary per the output
        of pyciemms sampling functions.
        timepoints (list): List of timepoints to associate with observations.
        label: If observations is a list, this is the label for the dataset.


    Returns -- Dataframe with:
       -  1st level index is the datsaet label
       -  2nd level index is a dataset sequence number
       -  Columns are timepoints
       -  Cell values are the data value
    """

    def _nice_type(values):
        if isinstance(values, torch.Tensor):
            values = values.numpy()
        return values

    if isinstance(timepoints, torch.Tensor):
        timepoints = timepoints.numpy()

    if isinstance(data, Union[list, torch.Tensor]):
        data = {label: np.array([v for v in data]).reshape((1, len(data)))}

    compatible = {k: v for k, v in data.items() if v.shape[-1] == len(timepoints)}

    dfs = {
        k: pd.DataFrame(data=_nice_type(v), columns=timepoints)
        for k, v in compatible.items()
    }

    df = pd.concat(dfs, names=["dataset", "instance"])
    df.columns.name = "time"
    return df


def trajectories(
    observations,
    *,
    points: Union[None, Any] = None,
    subset: Union[str, Callable, list] = all,
    qlow: float = 0.05,
    qhigh: float = 0.95,
    limit: Union[None, Integral] = None,
    colors: Dict[str, Any] = None,
    relabel: Union[None, Dict[str, str]] = None,
) -> VegaSchema:
    """_summary_

    TODO: Intervention marker line

    Args:
        observations (_type_): _description_
        points (_type_):
        subset (any, optional): Subset the 'observations' based on keys/values.
           - Default is the 'all' function, and it keeps all keys
           - If a string is present, it is treated as a regex and matched against the key
           - If a callable is present, it is called as f(key, value) and the key is kept for truthy values
           - Otherwise, assumed tob e a list-like of keys to keep
           If subset is specified, the color scale ordering follows the subset order.
        colors Dict[str, Any]: Color mapping from names to colors.
           Keys are pre-relabel names of data subsets (pre-relabel is used for simplicity of
           generating mappings from source-data).
           Mapping to 'None' or not including a key will drop the sequence (a fitler done
           in addition to the 'subset').
        relabel (None, Dict[str, str]): Relabel elements for rendering.
           If ommitted, relabling is the identity function.
        limit --
    """
    dataset_names = observations.index.get_level_values(0).unique()

    if subset == all:
        keep = dataset_names.keys()
    elif isinstance(subset, str):
        keep = [k for k in dataset_names if re.match(subset, k)]
    elif callable(subset):
        keep = [k for k, v in dataset_names if subset(k, v)]
    else:
        keep = subset

    observations = observations.loc[keep]

    if colors is not None:
        remaining_names = observations.index.get_level_values(0).unique()
        keep = [k for k in remaining_names if colors.get(k, None)]
        observations = observations.loc[keep]

    if relabel:
        rekeyed = [relabel.get(k, k) for k in observations.index.get_level_values(0)]
        observations = (
            observations.reset_index()
            .assign(dataset=rekeyed)
            .set_index(["dataset", "instance"])
        )

    tracks = observations.groupby(level=0).apply(
        lambda df: df if len(df) == 1 else None
    )
    return tracks

    track_ids = set(tracks.index.get_level_values(0).unique())
    not_track_ids = set(observations.index.get_level_values(0)).difference(track_ids)

    # compute quantiles
    not_tracks = observations.loc[not_track_ids]
    lows = not_tracks.groupby(level=0).apply(lambda df: df.quantile(q=qlow))
    highs = not_tracks.groupby(level=0).apply(lambda df: df.quantile(q=qhigh))
    dists = pd.concat({"lower": lows, "upper": highs})
    dists.index = dists.index.rename(("bound", "trajectory"))
    dists = dists[dists.columns[:limit]]
    dists = (
        dists.reset_index()
        .melt(
            value_vars=dists.columns,
            id_vars=["trajectory", "bound"],
        )
        .pivot(columns="bound", index=["trajectory", "time"])
        .droplevel(0, axis="columns")
        .reset_index()
        .to_dict(orient="records")
    )

    if len(tracks) > 0:
        tracks = (
            pd.DataFrame(tracks)
            .iloc[:limit]
            .melt(value_vars=tracks.keys(), id_vars="time")
            .rename(columns={"variable": "trajectory"})
            .to_dict(orient="records")
        )
    else:
        tracks = []

    if points is not None:
        points = (
            points.reset_index(level=0)
            .melt(id_vars=["dataset"])
            .iloc[:limit]
            .to_dict(orient="records")
        )
    else:
        points = []

    schema = _trajectory_schema()
    schema["data"] = replace_named_with(
        schema["data"], "distributions", ["values"], dists
    )
    schema["data"] = replace_named_with(schema["data"], "tracks", ["values"], tracks)
    schema["data"] = replace_named_with(schema["data"], "points", ["values"], points)

    if colors is not None:
        colors = {relabel.get(k, k): v for k, v in colors.items()}
        schema["scales"] = replace_named_with(
            schema["scales"], "color", ["domain"], [*colors.keys()]
        )
        schema["scales"] = replace_named_with(
            schema["scales"], "color", ["range"], [*colors.values]
        )

    return schema


# Things to check:
# _trajectories(prior_samples, tspan) == [*prior_samples.keys()]
# _trajectories(prior_samples, tspan, obs_keys = all) == [*prior_samples.keys()]
# _trajectories(prior_samples, tspan, obs_keys = ".*_sol") == ['Rabbits_sol', 'Wolves_sol']

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
    bins_count = sturges_bin(joint)
    counts, edges = np.histogram(joint, bins=bins_count)

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
