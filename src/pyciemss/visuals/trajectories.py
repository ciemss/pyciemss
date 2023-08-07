from typing import Dict, Union, Any, Optional

from numbers import Integral, Number
import pandas as pd
import numpy as np
import re

from . import vega


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


def clean_nans(
    entries: list[dict], *, parent: bool = True, replace: Optional[Any] = None
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


def trajectories(
    distributions: pd.DataFrame,
    *,
    traces: Optional[pd.DataFrame] = None,
    points: Optional[pd.DataFrame] = None,
    keep: Union[str, list, all] = all,
    drop: Union[str, list, None] = None,
    markers: Optional[dict[str, Number]] = None,
    relabel: Optional[Dict[str, str]] = None,
    colors: Optional[dict] = None,
    qlow: float = 0.05,
    qhigh: float = 0.95,
    join_points: bool = True,
    logy: bool = False,
    limit: Optional[Integral] = None,
    title: Optional[str] = None,
) -> vega.VegaSchema:
    """_summary_

    TODO: Handle the 'No distributions' case

    Args:
        observations (None, pd.DataFrame): Dataframe formatted per
        pyciemss.utils.interface_utils.convert_to_output_format
           These will be plotted as spans based on the qlow/qhigh parameters
        traces (None, pd.DataFrame): Example trajectories to plot.
        points (None, pd.DataFrame): Example points to plot (joined by lines)
        markers (None, list[Number]): Timepoint markers. Key is the label, value is the timepoint
        keep (str, list, all): Only keep some of the 'observations' based on keys/values.
           - Default is the 'all' function, and it keeps all columns
           - If a string is present, it is treated as a regex and matched against the columns. Matches are kept.
           - Otherwise, assumed to be a list-like of columns to keep
           If keep is specified, the color scale ordering follows the kept order.
        drop (str, list, None): Drop specific columns (applied AFTER keep)
          - Defaul is 'None', keeping all columns
          - If a string is present, it is treated as a regex and matched against the columns. Matches are dropped.
          - Otherwise, assumed to be a list-like of columns to drop
          Drop will not error if a name specified is not present at drop time.
        relabel (None, Dict[str, str]): Relabel elements for rendering.  Happens
            after key keep & drop.
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

    if keep == all:
        keep = distributions.columns
    elif isinstance(keep, str):
        keep = [k for k in distributions.columns if re.match(keep, k)]
    else:
        keep = keep

    if drop is None:
        drop = []
    elif isinstance(drop, str):
        drop = [k for k in distributions.columns if re.match(drop, k)]
    else:
        drop = drop

    distributions = (
        distributions.filter(items=keep)
        .drop(columns=drop, errors="ignore")
        .rename(columns=relabel)
    )

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

    schema = vega.load_schema("trajectories.vg.json")
    schema["data"] = vega.replace_named_with(
        schema["data"], "distributions", ["values"], clean_nans(distributions)
    )

    schema["data"] = vega.replace_named_with(
        schema["data"], "points", ["values"], clean_nans(points)
    )
    schema["data"] = vega.replace_named_with(
        schema["data"], "traces", ["values"], clean_nans(traces)
    )
    schema["data"] = vega.replace_named_with(
        schema["data"], "markers", ["values"], clean_nans(markers)
    )

    if colors is not None:
        colors = {k: v for k, v in colors.items() if k in all_trajectories}

        schema["scales"] = vega.replace_named_with(
            schema["scales"], "color", ["domain"], [*colors.keys()]
        )
        schema["scales"] = vega.replace_named_with(
            schema["scales"], "color", ["range"], [*colors.values()]
        )

    if not join_points:
        marks = vega.find_keyed(schema["marks"], "name", "_points")["marks"]
        simplified_marks = vega.delete_named(marks, "_points_line")
        schema["marks"] = vega.replace_named_with(
            schema["marks"], "_points", ["marks"], simplified_marks
        )
        schema

    if logy:
        schema = vega.rescale(schema, "yscale", "log")

    if title:
        schema = vega.set_title(schema, title)

    return schema


# Things to check:
# trajectories(prior_samples, tspan) == [*prior_samples.keys()]
# trajectories(prior_samples, tspan, obs_keys = all) == [*prior_samples.keys()]
# trajectories(prior_samples, tspan, obs_keys = ".*_sol") == ['Rabbits_sol', 'Wolves_sol']
# trajectories with and without join_points
# combinations of calls (colors, colors+relable, colors+subset, relable+subset, relabel+colors+subset, etc)
