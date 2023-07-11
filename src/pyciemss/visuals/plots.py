from typing import List, Dict, Any, Callable, Literal, Union
from collections.abc import Iterable

from numbers import Integral, Number

import pandas as pd
import numpy as np
import torch
import pkgutil
import json
import re
import os

from itertools import tee, filterfalse, compress
import IPython.display
from copy import deepcopy

import matplotlib.tri as tri
from pyro.distributions import Dirichlet


VegaSchema = Dict[str, Any]


def _histogram_multi_schema() -> VegaSchema:
    return json.loads(pkgutil.get_data(__name__, "histogram_static_bins_multi.vg.json"))


def _trajectory_schema() -> VegaSchema:
    return json.loads(pkgutil.get_data(__name__, "trajectories.vg.json"))


def _barycenter_triangle_schema() -> VegaSchema:
    return json.loads(pkgutil.get_data(__name__, "barycenter_triangle.vg.json"))


def _calibrate_schema() -> VegaSchema:
    return json.loads(pkgutil.get_data(__name__, "calibrate_chart.vg.json"))


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


def clean_nans(
    entries: list[dict], *, parent: bool = True, replace: Any = None
) -> list[dict]:
    """Clean list of entries for json serailizazation. ONLY LOOKS ONE LEVEL DOWN.
    Args:
        entries (list[dict]): Entries to clean (list of json dictionaries)
        parent (bool, optional): Replace parent (default) or key value?
        replace (Any, optional): Replace with this value.  If None, deletes.
    Returns:
        _type_: _description_
    """

    def is_nan(v):
        try:
            return np.isnan(v)
        except Exception:
            return False

    def has_nan(d):
        for v in d.values():
            if is_nan(v):
                return True
        return False

    def maybe_replace(d, replace_with):
        fix = [k for k, v in d.items() if is_nan(v)]
        for k in fix:
            d[k] = replace_with
        return d

    def remove_nan(d):
        fix = [k for k, v in d.items() if is_nan(v)]
        for k in fix:
            del d[k]
        return d

    if replace is None:
        if parent:
            entries = [e for e in entries if not has_nan(e)]
        else:
            entries = [remove_nan(e, replace) for e in entries]
    else:
        if parent:
            entries = [replace if has_nan(e) else e for e in entries]
        else:
            entries = [maybe_replace(e, replace) for e in entries]

    return entries


# Trajectory Visualizations ------------------


def nice_df(df):
    """Standardizes dataframe setup, imputing columns if possible and setting index."""

    if df.index.name is not None or df.index.names != [None]:
        df = df.reset_index()

    if "sample_id" not in df.columns:
        df = df.assign(sample_id=0)

    if "timepoint_id" not in df.columns:
        df = df.assign(timepoint_id=range(len(df)))

    timepoint_cols = [
        c for c in df.columns if c.startswith("timepoint_") and c != "timepoint_id"
    ]
    if len(timepoint_cols) == 0:
        df = df.assign(timepoint=df.timepoint_id)
    elif len(timepoint_cols) > 1:
        raise ValueError(
            f"Cannot work with multiple timepoint formats. Found {timepoint_cols}"
        )
    else:
        df = df.assign(timepoint=df[timepoint_cols[0]]).drop(columns=timepoint_cols)

    df = df.drop(columns=["timepoint_id"]).set_index(["timepoint", "sample_id"])
    return df


def trajectories(
    distributions: Union[None, pd.DataFrame] = None,
    *,
    traces: Union[None, pd.DataFrame] = None,
    points: Union[None, pd.DataFrame] = None,
    subset: Union[str, list] = all,
    markers: Union[None, dict[str, Number]] = None,
    relabel: Union[None, Dict[str, str]] = None,
    colors: Union[None, dict] = None,
    qlow: float = 0.05,
    qhigh: float = 0.95,
    join_points: bool = True,
    logy: bool = False,
    limit: Union[None, Integral] = None,
) -> VegaSchema:
    """_summary_

    TODO: Handle the 'No distributions' case

    Args:
        observations (None, pd.DataFrame): Dataframe formatted per
        pyciemss.utils.interface_utils.convert_to_output_format
           These will be plotted as spans based on the qlow/qhigh parameters
        traces (None, pd.DataFrame): Example trajectories to plot.
        points (None, pd.DataFrame): Example points to plot (joined by lines)
        markers (None, list[Number]): Timepoint markers. Key is the label, value is the timepoint
        subset (any, optional): Subset the 'observations' based on keys/values.
           - Default is the 'all' function, and it keeps all keys
           - If a string is present, it is treated as a regex and matched against the key
           - Otherwise, assumed tob e a list-like of keys to keep
           If subset is specified, the color scale ordering follows the subset order.
        relabel (None, Dict[str, str]): Relabel elements for rendering.  Happens
            after key subsetting.
        colors: Use the specified colors as a post-relable keyed dictionary to vega-valid color.
           Mapping to None or not includding a mapping will drop that sequence
        qlow (float): Lower percentile to use in obsersvation distributions
        qhigh (float): Higher percentile to use in obsersvation distributions
        join_points (bool): Should the plot of the points have a line joining through?
        limit (None, Integral) -- Only include up to limit number of records (mostly for debugging)
    """
    if relabel is None:
        relabel = dict()

    distributions = nice_df(distributions)
    if traces is not None:
        traces = nice_df(traces)
    if points is not None:
        points = nice_df(points)

    if subset == all:
        keep = distributions.columns
    elif isinstance(subset, str):
        keep = [k for k in distributions.columns if re.match(subset, k)]
    else:
        keep = subset

    distributions = distributions.filter(items=keep).rename(columns=relabel)

    if colors:
        keep = [k for k in distributions.columns if colors.get(k, None) is not None]
        distributions = distributions.filter(items=keep)

    point_trajectories = points.columns.tolist() if points is not None else []
    trace_trajectories = traces.columns.tolist() if traces is not None else []
    distributions_traj = (
        distributions.columns.tolist() if distributions is not None else []
    )
    all_trajectories = distributions_traj + trace_trajectories + point_trajectories

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
            .groupby(level=["trajectory", "timepoint"])
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
            points.melt(ignore_index=False, var_name="trajectory")
            .reset_index()
            .to_dict(orient="records")
        )
    else:
        points = []

    if markers is not None:
        markers = [{"timepoint": v, "label": k} for k, v in markers.items()]
    else:
        markers = []

    schema = _trajectory_schema()
    schema["data"] = replace_named_with(
        schema["data"], "distributions", ["values"], clean_nans(distributions)
    )

    schema["data"] = replace_named_with(
        schema["data"], "points", ["values"], clean_nans(points)
    )
    schema["data"] = replace_named_with(
        schema["data"], "traces", ["values"], clean_nans(traces)
    )
    schema["data"] = replace_named_with(
        schema["data"], "markers", ["values"], clean_nans(markers)
    )

    if colors is not None:
        colors = {k: v for k, v in colors.items() if k in all_trajectories}

        schema["scales"] = replace_named_with(
            schema["scales"], "color", ["domain"], [*colors.keys()]
        )
        schema["scales"] = replace_named_with(
            schema["scales"], "color", ["range"], [*colors.values()]
        )

    if not join_points:
        marks = find_keyed(schema["marks"], "name", "_points")["marks"]
        simplified_marks = delete_named(marks, "_points_line")
        schema["marks"] = replace_named_with(
            schema["marks"], "_points", ["marks"], simplified_marks
        )
        schema

    if logy:
        schema = rescale(schema, "yscale", "log")

    return schema


# Things to check:
# trajectories(prior_samples, tspan) == [*prior_samples.keys()]
# trajectories(prior_samples, tspan, obs_keys = all) == [*prior_samples.keys()]
# trajectories(prior_samples, tspan, obs_keys = ".*_sol") == ['Rabbits_sol', 'Wolves_sol']
# trajectories with and without join_points
# combinations of calls (colors, colors+relable, colors+subset, relable+subset, relabel+colors+subset, etc)


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


# ---------


def triangle_weights(samples, concentration=20, subdiv=7):
    # Adapted from https://blog.bogatron.net/blog/2014/02/02/visualizing-dirichlet-distributions/
    # TODO: This method works...but it quite the monstrosity!  Look into ways to simplify...

    AREA = 0.5 * 1 * 0.75**0.5

    def _tri_area(xy, pair):
        return 0.5 * np.linalg.norm(np.cross(*(pair - xy)))

    def _xy2bc(xy, tol=1.0e-4):
        """Converts 2D Cartesian coordinates to barycentric."""
        coords = np.array([_tri_area(xy, p) for p in pairs]) / AREA
        return np.clip(coords, tol, 1.0 - tol)

    corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)

    # For each corner of the triangle, the pair of other corners
    pairs = [corners[np.roll(range(3), -i)[1:]] for i in range(3)]
    # The area of the triangle formed by point xy and another pair or points

    # convert to coordinates with 3, rather than to points of reference for Direichlet input
    points = torch.tensor(np.array([(_xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]))
    points /= torch.sum(points, dim=1, keepdim=True)

    alpha = samples * concentration
    vals = torch.stack(
        [
            torch.exp(Dirichlet(alpha).log_prob(points[i, :]))
            for i in range(points.shape[0])
        ]
    )
    vals /= torch.max(vals, dim=0, keepdim=True)[0]
    vals = torch.sum(vals, dim=1)
    vals /= torch.sum(vals)

    coordinates_dict = {}

    # skip every line as alternates half of each lines
    y_num = 0
    not_use_trimesh_y = []
    for y in np.unique(trimesh.y):
        y_num += 1
        if y_num % 2 == 0:
            not_use_trimesh_y.append(y)

    df_coord = pd.DataFrame({"x": trimesh.x, "y": trimesh.y, "z": vals.tolist()})
    not_use_trimesh_x = list(
        np.unique(df_coord[df_coord.y == not_use_trimesh_y[0]]["x"].tolist())
    )

    # save all existing coordinates
    for x, y, z in zip(trimesh.x, trimesh.y, vals):
        coordinates_dict[(x, y)] = z.item()

    # fill in missing part of square grid
    for x in np.unique(trimesh.x):
        for y in np.unique(trimesh.y):
            if (x, y) not in coordinates_dict.keys():
                coordinates_dict[x, y] = 0

    # convert to dataframe and sort with y first in descending order
    df = pd.DataFrame(coordinates_dict.items(), columns=["x,y", "val"])
    df[["x", "y"]] = pd.DataFrame(df["x,y"].tolist(), index=df.index)
    df = df.sort_values(["y", "x"], ascending=[False, True])

    # remove the alternative values, (every other y and all the values associated with that y)
    df_use = df[(~df.x.isin(not_use_trimesh_x)) & (~df.y.isin(not_use_trimesh_y))]

    json_dict = {}
    json_dict["width"] = len(np.unique(df_use.x))
    json_dict["height"] = len(np.unique(df_use.y))
    json_dict["values"] = df_use["val"].tolist()

    return json_dict


def calibration(datasource: pd.DataFrame):
    """Create a contour plot from the passed datasource.

    datasource --  A dataframe ready for rendering.  Should include:
       - time (int)
       - column_names (str)
       - calibration (bool)
       - y  --- Will be shown as a line
       - y1 --- Upper range of values
       - y0 --- Lower range of values
    """
    schema = _calibrate_schema()

    data = find_keyed(schema["data"], "name", "table")
    del data["url"]
    data["values"] = datasource.to_dict(orient="records")

    options = sorted(datasource["column_names"].unique().tolist())
    var_filter = find_keyed(schema["signals"], "name", "Variable")
    var_filter["bind"]["options"] = options
    var_filter["value"] = options[0]

    return schema


def triangle_contour(data, *, title=None, contour=True):
    """Create a contour plot from the passed datasource.

    datasource --
      * filename: File to load data from that will be loaded via vega's "url" facility

                  Path should be relative to the running file-server, as they will be
                  resolved in that context. If in a notebook, it is relative to the notebook
                  (not the root notebook server processes).
      * dataframe: A dataframe ready for rendering.  The data will be inserted into the schema
                as a record-oriented dictionary.

    kwargs -- If passing filename, extra parameters to the vega's url facility

    """
    mesh_data = triangle_weights(data)

    schema = _barycenter_triangle_schema()
    schema["data"] = replace_named_with(
        schema["data"],
        "contributions",
        ["values"],
        mesh_data,
    )

    if title:
        schema["title"] = title

    if not contour:
        contours = find_keyed(schema["marks"], "name", "_contours")
        contours["encode"]["enter"]["stroke"] = {
            "scale": "color",
            "field": "contour.value",
        }

    return schema


# -------- Utlity functions for working with Vega plots


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


def rescale(
    schema: VegaSchema,
    scale_name: str,
    scaletype: str,
    zero: Literal["auto", True, False] = "auto",
) -> VegaSchema:
    """Change a scale to a new interpolation type.

    Args:
        schema (VegaSchema): Vega schema to modify
        scale_name (str): _description_. Defaults to "yscale".
        scaletype (str): _description_. Defaults to "log".
        zero (Literal["auto", True, False]): "Auto" sets to 'true' if non-log scale.

    Returns:
        VegaSchema: Copy of the schema with an updated scale
    """
    schema = deepcopy(schema)

    if zero == "auto":
        zero = scaletype not in ["log", "symlog"]

    schema["scales"] = replace_named_with(
        schema["scales"], scale_name, ["type"], scaletype
    )

    schema["scales"] = replace_named_with(schema["scales"], scale_name, ["zero"], zero)

    return schema


def title(schema, title: str, *, target: Literal[None, "x", "y"] = None):
    schema = deepcopy(schema)
    if target is None:
        schema["title"] = title
    elif target == "x":
        axis = find_keyed(schema["axes"], "name", "x_axis")
        axis["title"] = title
    elif target == "y":
        axis = find_keyed(schema["axes"], "name", "y_axis")
        axis["title"] = title

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
    if qty is None:
        if "padding" in schema:
            del schema["padding"]
    else:
        schema["padding"] = qty

    return schema


def find_keyed(ls: list[dict], key: str, value: Any):
    """In the list of dicts, finds a think where key=value"""
    for e in ls:
        try:
            if e[key] == value:
                return e
        except Exception:
            pass

    raise ValueError(f"Attempted to find, but {key}={value} not found.")


def delete_named(ls: list, name: str) -> List:
    "REMOVE the first thing in ls where 'name'==name"

    def _maybe_keep(e, found):
        if found[0]:
            return True

        if e.get("name", None) == name:
            found[0] = True
            return False
        return True

    found = [False]
    filter = [_maybe_keep(e, found) for e in ls]
    if not found[0]:
        raise ValueError(f"Attempted to remove, but {name=} not found.")

    return [*compress(ls, filter)]


def orient_legend(schema: VegaSchema, name: str, location: Union[None, str]):
    schema = deepcopy(schema)
    legend = find_keyed(schema["legends"], "name", name)

    if location is None:
        if "orient" in legend:
            del legend["orient"]
    else:
        legend["orient"] = location

    return schema


def replace_named_with(ls: list, name: str, path: List[Any], new_value: Any) -> List:
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
            if path:
                for step in path[:-1]:
                    part = part[step]
                part[path[-1]] = new_value
            else:
                e = new_value

            done_replacing[0] = True

        return e

    done_replacing = [False]
    updated = [_maybe_replace(e, done_replacing) for e in ls]

    if not done_replacing[0]:
        raise ValueError(f"Attempted to replace, but {name=} not found.")

    return updated
