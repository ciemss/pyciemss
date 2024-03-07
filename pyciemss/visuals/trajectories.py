import re
from numbers import Integral, Number
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from . import vega


def select_traces(
    traces,
    *,
    select_by: Literal["mean", "var", "granger"] = "mean",
    keep: Union[str, list, Literal["all"]] = "all",
    drop: Union[str, list, None] = None,
    relabel: Optional[Dict[str, str]] = None,
):
    """Picks an actual trajectory based on the envelope of trajectories and a selection criteria

    Args:
        traces: Dataframe of traces
        subset: only keep subset columns
        select_by -- Method for selecting a trajectory
          - "mean" -- Trajectory closest to the mean-line of the envelope of all trajectories
          - "var" -- Trajectory that has the most-similar dynamics to the mean-line of the envelope of all trajectories
          - "granger" -- Trajectory that "best predicts" the mean-line of the envelope
                        (by the grangercausalitytest, maxlag=10)

        keep (str, list, "all"): Only keep some of the 'distributions' based on keys/values.
           - Default is the "all" string and it keeps all columns
           - If any other string is present, it is treated as a regex and matched against the columns. Matches are kept.
           - Otherwise, assumed to be a list-like of columns to keep
           If keep is specified, the color scale ordering follows the kept order.
        drop (str, list, None): Drop specific columns from 'distributions' (applied AFTER keep)
          - Default is 'None', keeping all columns
          - If a string is present, it is treated as a regex and matched against the columns. Matches are dropped.
          - Otherwise, assumed to be a list-like of columns to drop
          Drop will not error if a name specified is not present at drop time.
        relabel (None, Dict[str, str]): Relabel elements for rendering.  Happens
            after keep & drop.

        #TODO: Add a way to control the time-axis.  _nice_df does some heuristics BUT we should provide an override
    """
    traces_df = _nice_df(traces)
    traces_df = _keep_drop_rename(traces_df, keep, drop, relabel)

    # get average line
    examplary_line_df = (
        traces_df.melt(ignore_index=False, var_name="trajectory")
        .set_index("trajectory", append=True)
        .groupby(level=["trajectory", "timepoint"])
        .mean()
        .reset_index()
    )

    def granger_fun(x):
        # first column is the column to compare to
        # not sure what maxlag value to use
        # return ssr-based-F test p value
        granger_value = grangercausalitytests(x[["mean_value", "value"]], maxlag=[10])[
            10
        ][0]["ssr_ftest"][1]
        return granger_value

    # all data, melted
    melt_all = traces_df.melt(ignore_index=False, var_name="trajectory").reset_index()
    # add value of average line to original df as column
    merged_all_mean = pd.merge(
        melt_all, examplary_line_df, on=["trajectory", "timepoint"]
    ).rename(columns={"value_x": "value", "value_y": "mean_value"})
    # get distance from average line for each sample/timepoint/trajectory
    merged_all_mean["distance_mean"] = abs(
        merged_all_mean["mean_value"] - merged_all_mean["value"]
    )
    # get sum of distance from mean for all timepoints in sample/trajectory
    group_examplary = merged_all_mean.set_index(
        ["trajectory", "timepoint", "sample_id"], append=True
    ).groupby(level=["trajectory", "sample_id"])

    # get sum and variable of each trajectory (rabbit or wolf) and sample id group
    if select_by == "mean":
        sum_examplary = (
            group_examplary.mean().reset_index()
        )  # get only min distance from each trajectory type (rabbit or worlf) back
        best_sample_id = sum_examplary.loc[
            sum_examplary.groupby("trajectory").distance_mean.idxmin()
        ][["sample_id", "trajectory"]]

    elif select_by == "var":
        sum_examplary = (
            group_examplary.var().reset_index()
        )  # get only min distance from each trajectory type (rabbit or worlf) back
        best_sample_id = sum_examplary.loc[
            sum_examplary.groupby("trajectory").distance_mean.idxmin()
        ][["sample_id", "trajectory"]]

    elif select_by == "granger":
        granger_examplary = group_examplary.apply(lambda x: granger_fun(x))
        sum_examplary = pd.DataFrame({"granger": granger_examplary})
        sum_examplary = sum_examplary.reset_index()
        best_sample_id = sum_examplary.loc[
            sum_examplary.groupby("trajectory").granger.idxmin()
        ][["sample_id", "trajectory"]]

    # only keep sample id's from 'best' lines
    only_examplary_line = pd.merge(
        melt_all, best_sample_id, on=["sample_id", "trajectory"], how="right"
    )

    # get into dataframe in correct foramt for trajectories traces argument
    only_examplary_line = only_examplary_line.drop(
        columns=["sample_id"], errors="ignore"
    )
    examplary_line = only_examplary_line.pivot_table(
        values="value", index="timepoint", columns="trajectory"
    )
    examplary_line["timepoint_id"] = examplary_line.index
    return examplary_line


def trajectories(
    distributions: pd.DataFrame,
    *,
    traces: Optional[pd.DataFrame] = None,
    points: Optional[pd.DataFrame] = None,
    keep: Union[str, list, Literal["all"]] = "all",
    drop: Union[str, list, None] = None,
    base_markers_v: Optional[Dict[str, Number]] = None,
    base_markers_h: Optional[Dict[str, Number]] = None,
    relabel: Optional[Dict[str, str]] = None,
    colors: Optional[Dict] = None,
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
        distributions (pd.DataFrame): Dataframe formatted per
        pyciemss.utils.interface_utils.convert_to_output_format
           These will be plotted as spans based on the qlow/qhigh parameters
        traces (None, pd.DataFrame): Example trajectories to plot.
        points (None, pd.DataFrame): Example points to plot (joined by lines)
        base_markers (None, Dict[str, Number]): Timepoint markers. Key is the label, value is the timepoint
        base_markers_h (None, Dict[str, Number]): Horizontal markers. Key is the label, value is the horizonal value
        keep (str, list, "all"): Only keep some of the 'distributions' based on keys/values.
           - Default is the string "all", and it keeps all columns
           - If a any other string is present, it is treated as a regex and matched against the columns.
             Matches are kept.
           - Otherwise, assumed to be a list-like of columns to keep
           If keep is specified, the color scale ordering follows the kept order.
        drop (str, list, None): Drop specific columns from 'distributions' (applied AFTER keep)
          - Default is 'None', keeping all columns
          - If a string is present, it is treated as a regex and matched against the columns. Matches are dropped.
          - Otherwise, assumed to be a list-like of columns to drop
          Drop will not error if a name specified is not present at drop time.
        relabel (None, Dict[str, str]): Relabel elements for rendering.  Happens
            after keep & drop.
        colors: Use the specified colors as a post-relable keyed dictionary to vega-valid color.
           Mapping to None or not includding a mapping will drop that sequence
        qlow (float): Lower percentile to use in obsersvation distributions
        qhigh (float): Higher percentile to use in obsersvation distributions
        join_points (bool): Should the plot of the points have a line joining through?
        limit (None, Integral) -- Only include up to limit number of records (mostly for debugging)
    """
    if relabel is None:
        relabel = dict()

    distributions = _nice_df(distributions)
    traces = _nice_df(traces)
    points = _nice_df(points)

    distributions = _keep_drop_rename(distributions, keep, drop, relabel)

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

    if base_markers_v is not None:
        markers_v = [{"axis_value": v, "label": k} for k, v in base_markers_v.items()]
    else:
        markers_v = []

    if base_markers_h is not None:
        markers_h = [{"axis_value": v, "label": k} for k, v in base_markers_h.items()]
    else:
        markers_h = []

    schema = vega.load_schema("trajectories.vg.json")
    schema["data"] = vega.replace_named_with(
        schema["data"], "distributions", ["values"], _clean_nans(distributions)
    )

    schema["data"] = vega.replace_named_with(
        schema["data"], "points", ["values"], _clean_nans(points)
    )
    schema["data"] = vega.replace_named_with(
        schema["data"], "traces", ["values"], _clean_nans(traces)
    )
    schema["data"] = vega.replace_named_with(
        schema["data"], "markers_v", ["values"], _clean_nans(markers_v)
    )
    schema["data"] = vega.replace_named_with(
        schema["data"], "markers_h", ["values"], _clean_nans(markers_h)
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


# --------------- Shared utilities


def _nice_df(df):
    """Internal utility.  Standardizes dataframe setup, imputing columns if possible and setting index."""
    if df is None:
        return df

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


def _clean_nans(
    entries: List[Dict], *, parent: bool = True, replace: Optional[Any] = None
) -> List[Dict]:
    """Internal Utlitiy. Clean list of entries for json serailizazation.
    ONLY LOOKS ONE LEVEL DOWN.

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
            entries = [maybe_replace(e, replace) for e in entries]
    else:
        if parent:
            entries = [replace if has_nan(e) else e for e in entries]
        else:
            entries = [maybe_replace(e, replace) for e in entries]

    return entries


def _keep_drop_rename(df, keep, drop, relabel):
    """
    Internal utility: Implements consistent keep/drop/rename logic
    for use across several utilities
    """

    if keep == "all":
        keep = df.columns
    elif isinstance(keep, str):
        try:
            m = re.compile(keep)
        except Exception as e:
            raise ValueError(f"Error compiling 'keep' expression: {str(e)}")

        keep = [c for c in df.columns if m.match(c)]
    else:
        keep = keep

    if drop is None:
        drop = []
    elif isinstance(drop, str):
        try:
            m = re.compile(drop)
        except Exception as e:
            raise ValueError(f"Error compiling 'drop' expression: {str(e)}")

        drop = [c for c in df.columns if m.match(c)]
    else:
        drop = drop

    relabel = {} if relabel is None else relabel

    return (
        df.filter(items=keep)
        .drop(columns=drop, errors="ignore")
        .rename(columns=relabel)
    )
