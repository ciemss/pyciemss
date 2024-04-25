import re
from numbers import Integral, Number
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler
from . import vega

def get_examplary_lines(traces_df, kmean = False, n_clusters =4):

    def return_kmeans(traces_df, n_clusters =4):
        # get the trajectory for current trajectory
        dba_km = TimeSeriesKMeans(n_clusters=n_clusters,
                          n_init=2,
                          metric="dtw",
                          verbose=True,
                          max_iter_barycenter=10)
        
        traces_df_new = traces_df.join(
            traces_df.groupby('sample_id')[['value']]
                    .transform('mean').rename(columns={'value': 'sample_mean'})
            )
        # traces_df_new  = traces_df_new.rename(columns={'value': 'orig_value'})

        traces_df_new["norm_value"] = traces_df_new['value'] - traces_df_new['sample_mean']
        #  for timeseries kmeans each row is a time series, each column is a new time point
        # in my sin function, each column was a time series, and each row is time point 
        traces_df_pivot_norm = traces_df_new[['sample_id', "norm_value", "timepoint"]].pivot_table(
        values="norm_value", index = "sample_id", columns="timepoint"
    )
        traces_df_pivot = traces_df_new[['sample_id', "value", "timepoint"]].pivot_table(
        values="value", index = "sample_id", columns="timepoint"
    )
    
        y_pred = dba_km.fit_predict(traces_df_pivot_norm)
        cluster_sample_id = {}
        all_clusters = []
        for yi in range(n_clusters):
            traces_df_T_cluster = traces_df_pivot[y_pred == yi]
            # get mean per timepoint of the values
            means_trajectory = (
                traces_df_T_cluster.melt(ignore_index=False, var_name="timepoint")
                .reset_index()
                .set_index('timepoint')['value']
                .groupby(level=["timepoint"])
                .mean()
                .reset_index()
            )
            means_trajectory["cluster"] = "cluster_" + str(yi)
            cluster_sample_id["cluster_" + str(yi)] = list(np.unique(traces_df_T_cluster.index))
            all_clusters.append(means_trajectory[['timepoint', 'value', "cluster"]])

        all_clusters_df = pd.concat(all_clusters)
        return all_clusters_df, cluster_sample_id

    if kmean:
        all_trajectories = []
        # melts so trajectories (D_state, infected_state, etc.) are one column of trajectory times and one of values
        traces_df_melt = traces_df.melt(ignore_index=False, var_name="trajectory").set_index("trajectory", append=True).reset_index()
        # get mean per timepoint for each trajectory and cluster in dataframe (all_trajectoryies_df)
        cluster_sample_ids = {}
        for trajectory in np.unique(traces_df_melt['trajectory']):
            current_trajectory = traces_df_melt[traces_df_melt['trajectory'] == trajectory]
            all_clusters, cluster_sample_id = return_kmeans(current_trajectory, n_clusters=n_clusters)
            all_clusters['trajectory'] = trajectory
            all_trajectories.append(all_clusters)
            cluster_sample_ids[trajectory] = cluster_sample_id
        all_trajectories_df = pd.concat(all_trajectories)

        # separate by cluster so can use cluster group for mean comparison
        cluster_trajectory = {}
        
        for trajectory in np.unique(traces_df_melt['trajectory']):
            all_means = {}
            for cluster in np.unique(all_trajectories_df['cluster']):
                traces_df_cluster = all_trajectories_df[(all_trajectories_df['cluster'] == cluster) & (all_trajectories_df['trajectory'] == trajectory)]
                all_means[cluster] = {'cluster_mean': traces_df_cluster, 'cluster_sample_id': cluster_sample_ids[trajectory][cluster]}
            cluster_trajectory[trajectory] = all_means

    else:
        cluster_trajectory ={}
        # get average line, 3 columns (trajectory: "D_state", timepoint, value)
        means_trajectory = (
            traces_df.melt(ignore_index=False, var_name="trajectory")
            .set_index("trajectory", append=True)
            .groupby(level=["trajectory", "timepoint"])
            .mean()
            .reset_index()
        )
        for trajectory in np.unique(means_trajectory['trajectory']):
            traces_df_traj= means_trajectory[(means_trajectory['trajectory'] == trajectory)]
            traces_df = traces_df.reset_index()
            cluster_trajectory[trajectory] = {'no_cluster': {'cluster_mean': traces_df_traj, 'cluster_sample_id': list(np.unique(traces_df['sample_id']))}}
    return cluster_trajectory

def grouped_mean(traces_df, means_trajectory):


    # all data, melted
    melt_all = traces_df.melt(ignore_index=False, var_name="trajectory").reset_index()
    # add value of average line to original df as column
    merged_all_mean = pd.merge(
        melt_all, means_trajectory, on=["trajectory", "timepoint"]
    ).rename(columns={"value_x": "value", "value_y": "mean_value"})
    # get distance from average line for each sample/timepoint/trajectory
    merged_all_mean["distance_mean"] = abs(
        merged_all_mean["mean_value"] - merged_all_mean["value"]
    )
    
    # get sum of distance from mean for all timepoints in sample/trajectory
    group_examplary = merged_all_mean.set_index(
        ["trajectory", "timepoint", "sample_id"], append=True
    ).groupby(level=["trajectory", "sample_id"])
    return melt_all, group_examplary


def grouped_mean(traces_df, means_trajectory):


    # all data, melted so each parameter column becomes a column called trajectory
    melt_all = traces_df.melt(ignore_index=False, var_name="trajectory").reset_index()
    # add value of average line to original df as column
    merged_all_mean = pd.merge(
        melt_all, means_trajectory, on=["trajectory", "timepoint"]
    ).rename(columns={"value_x": "value", "value_y": "mean_value"})
    # get distance from average line for each sample/timepoint/trajectory
    merged_all_mean["distance_mean"] = abs(
        merged_all_mean["mean_value"] - merged_all_mean["value"]
    )
    
    # get sum of distance from mean for all timepoints in sample/trajectory
    group_examplary = merged_all_mean.set_index(
        ["trajectory", "timepoint", "sample_id"], append=True
    ).groupby(level=["trajectory", "sample_id"])
    return melt_all, group_examplary

## TODO return ids for multiple trajectories
def trajectory_chaos(trajectories: pd.DataFrame):
    """Consecutive points comparison"""
    def order(g):
        labels = g.reset_index(level=1).apply(lambda r: f"{r['trajectory']}_{r['sample_id']}", axis="columns")
        ordered_labels = g.assign(label=labels.values).sort_values("value")["label"].values
        return ordered_labels
    
    trajectories = trajectories.melt(ignore_index=False, var_name="trajectory")
    orders = trajectories.groupby(level=0).apply(order)
    order_dict = {}
    for current_timepoint in orders:
        i = 0
        for sample_trajectory in current_timepoint:
            if sample_trajectory not in order_dict:
                order_dict[sample_trajectory] = []
                # record order (rank) that this sample_trajectory turned up at the current timepoint
                order_dict[sample_trajectory].append(i)
            else:
                order_dict[sample_trajectory].append(i)
            i += 1

    current_max = -1
    for sample_trajectory_key, sample_trajectory_value in order_dict.items():
        # get difference in ranking
        sum_difference = np.sum([abs(j-i) for i, j in zip(sample_trajectory_value[:-1], sample_trajectory_value[1:])])
        if sum_difference > current_max:
            max_chaos_sample_id = int(float(sample_trajectory_key.split('_')[-1]))
            current_max = sum_difference
    trajectories = trajectories.reset_index()
    best_sample_id = trajectories[trajectories['sample_id'] == max_chaos_sample_id][["sample_id", "trajectory"]]
    return best_sample_id

def get_best_example(group_examplary, select_by):
    """Picks an actual trajectory based on the envelope of trajectories and a selection criteria

    Args:
        select_by -- Method for selecting a trajectory
          - "mean" -- Trajectory closest to the mean-line of the envelope of all trajectories
          - "var" -- Trajectory that has the most-similar dynamics to the mean-line of the envelope of all trajectories
          - "granger" -- Trajectory that "best predicts" the mean-line of the envelope
                        (by the grangercausalitytest, maxlag=10)
    """
    def granger_fun(x):
        # first column is the column to compare to
        # not sure what maxlag value to use
        # return ssr-based-F test p value
        try:
            granger_value = grangercausalitytests(x[["mean_value", "value"]], maxlag=[10])[
                10
            ][0]["ssr_ftest"][1]
            return granger_value
        except:
            return np.nan
  
    # get sum and variable of each trajectory (rabbit or wolf) and sample id group
    if select_by == "mean":
        # get average distance from the mean per trajectory and sample id
        sum_examplary = (
            group_examplary['distance_mean'].mean().reset_index()
        )  # get only min distance from each trajectory type (rabbit or wolf) back
        best_sample_id = sum_examplary.loc[
            sum_examplary.groupby("trajectory").distance_mean.idxmin()
        ][["sample_id", "trajectory"]]

    elif select_by == "var":
        # get the variance per trajectory/sample id of difference from the mean (saved as distance_mean)
        sum_examplary = (
            group_examplary['distance_mean'].var().reset_index()
        )  # get only min distance from each trajectory type (rabbit or worlf) back
        best_sample_id = sum_examplary.loc[
            sum_examplary.groupby("trajectory").distance_mean.idxmin()
        ][["sample_id", "trajectory"]]

    elif select_by == "granger":
        granger_examplary = group_examplary.apply(lambda x: granger_fun(x))
        sum_examplary = pd.DataFrame({"granger": granger_examplary})
        sum_examplary = sum_examplary.reset_index()
        # return the sample id with the lowest granger significance score
        best_sample_id = sum_examplary.loc[
            sum_examplary.groupby("trajectory").granger.idxmin()
        ][["sample_id", "trajectory"]]
    return best_sample_id


def convert_back_trace_format(only_examplary_line):
        # get into dataframe in correct foramt for trajectories traces argument
    # remove sample_id because only have one line per trajectory
    only_examplary_line = only_examplary_line.drop(
        columns=["sample_id"], errors="ignore"
    )
    # back so each column is a trajectory type (liek *_state)
    examplary_line = only_examplary_line.pivot_table(
        values="value", index="timepoint", columns="trajectory"
    )
    return examplary_line

def convert_examplary_line(melt_all, best_sample_id):
    """Picks an actual trajectory based on the envelope of trajectories and a selection criteria
    """
    # only keep sample id's from 'best' lines
    only_examplary_line = pd.merge(
        melt_all, best_sample_id, on=["sample_id", "trajectory"], how="right"
    )
    # get into dataframe in correct foramt for trajectories traces argument
    # remove sample_id because only have one line per trajectory
    examplary_line = convert_back_trace_format(only_examplary_line)
    return examplary_line

def select_traces(
    traces,
    *,
    select_by_list: list = ['mean', 'var', 'granger', 'chaos'],
    keep: Union[str, list, Literal["all"]] = "all",
    drop: Union[str, list, None] = None,
    relabel: Optional[Dict[str, str]] = None,
    kmean: bool = False
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

    """
    

    traces_df = _nice_df(traces)
    traces_df = _keep_drop_rename(traces_df, keep, drop, relabel)
    # get mean (or kmeans) lines
    means_trajectory = get_examplary_lines(traces_df, kmean)
    i = 0
    examplary_line_list = []
    mean_line_list = []
    for select_by in select_by_list: 
        for (trajectory_key, trajectory_dict) in means_trajectory.items():
            for cluster_key in trajectory_dict.keys():
                mean_trajectory= trajectory_dict[cluster_key]['cluster_mean']
                cluster_sample_id = trajectory_dict[cluster_key]['cluster_sample_id']
                i +=1
                # keep only row with the right sample ids and colums from that trajectory keys
                traces_df_sample_id = traces_df[traces_df.index.get_level_values('sample_id').isin(cluster_sample_id)].loc[:,[trajectory_key]]


                # get grouped difference from the mean
                melt_all, group_examplary = grouped_mean(traces_df_sample_id, mean_trajectory)
                if select_by == "chaos":
                    best_sample_id = trajectory_chaos(traces_df_sample_id)
                else:
                    # get id of the sample with the lowest variance, differnce of granger socre
                    best_sample_id = get_best_example(group_examplary, select_by)
                # get examplar line
                examplary_line = convert_examplary_line(melt_all, best_sample_id)
                mean_trajectory = convert_back_trace_format(mean_trajectory)
                # if kmeans want to keep all 
                examplary_df = pd.DataFrame({"examplary_line": examplary_line.iloc[:,0], "mean_trajectory": mean_trajectory.iloc[:,0]})
                examplary_df['sample_id'] = best_sample_id['sample_id'].values[0]
                examplary_df['cluster'] = cluster_key.title()
                examplary_df['trajectory'] = trajectory_key.title()
                examplary_df['select_by'] = select_by.title()

                examplary_line_list.append(examplary_df)

    examplary_line_df = pd.concat(examplary_line_list, axis = 0)
    # examplary_line_df["timepoint_id"] = examplary_line_df.index 
    # mean_line_df["timepoint_id"] = mean_line_df.index 
    # return mean line per trajectory/cluster (np.unique(mean_trajecotry['cluster])), examplary_line, and other metrics, trajectory, best_examplar selected by
    return examplary_line_df



def trajectories(
    distributions: pd.DataFrame,
    *,
    traces: Optional[pd.DataFrame] = None,
    points: Optional[pd.DataFrame] = None,
    keep: Union[str, list, Literal["all"]] = "all",
    drop: Union[str, list, None] = None,
    base_markers: Optional[Dict[str, Number]] = None,
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
        markers (None, list[Number]): Timepoint markers. Key is the label, value is the timepoint
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
        traces_melt = (
            traces.melt(ignore_index=False, var_name="trajectory")
            .reset_index()
        )
        traces_melt['color_by'] = traces_melt['trajectory'].apply(lambda x: " ".join(x.split('_')[-2:]))
        traces = traces_melt.to_dict(orient="records")
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

    if base_markers is not None:
        markers = [{"timepoint": v, "label": k} for k, v in base_markers.items()]
    else:
        markers = []

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
        schema["data"], "markers", ["values"], _clean_nans(markers)
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
