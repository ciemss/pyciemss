from typing import List, Dict, Any, Callable
from numbers import Number

from copy import deepcopy
import IPython.display
import pandas as pd
import numpy as np
import torch

import pkgutil
import json

import re

from itertools import tee, filterfalse, chain


def _histogram_multi_schema():
    return json.loads(pkgutil.get_data(__name__, "histogram_static_bins_multi.vg.json"))

def _trajectory_schema():
    return json.loads(pkgutil.get_data(__name__, "trajectories.vg.json"))

# General Utilities ---------------

def partition(pred, iterable):
    t1, t2 = tee(iterable)
    return filterfalse(pred, t1), filter(pred, t2)


def tensor_dump(tensors, path):
    reformatted = {k: v.detach().numpy().tolist() 
               for k,v in tensors.items()}

    with open(path, "w") as f:
        json.dump(reformatted, f)


def tensor_load(path):
    with open(path) as f:
        data = json.load(f)
        
    data = {k: torch.from_numpy(np.array(v)) 
            for k, v in data.items()}
    
    return data


# Trajectory Visualizations ------------------

def _quantiles(values, qlow, qhigh):
    """Compute quantiles from torch tensors along the 0 dimension
    """
    low = torch.quantile(values, qlow, dim=0).detach().numpy()
    high = torch.quantile(values, qhigh, dim=0).detach().numpy()
    return {"lower": low, "upper": high}


def trajectories(observations, 
                      tspan, 
                      *,
                      obs_keys = all,
                      qlow = 0.05,
                      qhigh = 0.95,
                      limit=None):
    """_summary_

    TODO: Interpolation method probably needs attention...

    Args:
        observations (_type_): _description_
        tspan (_type_): _description_
        obs_keys (any, optional): Subset the 'observations' based on keys/values.
           - Default is the 'all' function, and it keeps all keys
           - If a string is present, it is treated as a regex and matched against the key
           - If a callable is present, it is called as f(key, value) and the key is kept for truthy values
           - Otherwise, assumed tob e a list-like of keys to keep
    """
    if obs_keys == all:
        keep = observations.keys()
    elif isinstance(obs_keys , str):
        keep = [k for k in observations.keys()
                if re.match(obs_keys , k)]
    elif callable(obs_keys):
        keep = [k for k, v in observations.items()
                if obs_keys(k,v)]
    else:
        keep = obs_keys 
        
    observations = {k: v for k,v in observations.items() 
                    if k in keep}
    
    exact = {k: v for k,v in observations.items() 
              if len(v.shape) == 1}            

    ranges = {k: _quantiles(v, qlow, qhigh) 
              for k,v in observations.items()
              if k not in exact}
    
    for title, values in exact.items():
        print("NOT IMPLEMENTED: Single trajectory")
        #Data poitns were handled by partial string-matching keys from data and observations before...
        #  That implicit mechanism is fraught.  An explicit mapping or cross-matching function might be 
        #  more useful over time
        
        break
    
    
    dfs = [pd.DataFrame.from_dict(ranges[k]).assign(trajectory=k, time=tspan)
            for k in ranges.keys()]

    dataset = [*chain.from_iterable(d.iloc[:limit].to_dict(orient="records") for d in dfs)]

    schema = _trajectory_schema()
    schema["data"][0] = {"name": "table", "values": dataset}
    
    return schema


## Things to check:
# _trajectories(prior_samples, tspan) == [*prior_samples.keys()]
# _trajectories(prior_samples, tspan, obs_keys = all) == [*prior_samples.keys()]
# _trajectories(prior_samples, tspan, obs_keys = ".*_sol") == ['Rabbits_sol', 'Wolves_sol']

## Called like:
# plot = vega.trajectories(prior_samples, tspan, obs_keys=".*_sol")
# with open("trajectories.json", "w") as f:
#     json.dump(plot, f, indent=3)
# vega.ipy_display(plot)


## Histogram visualizations ------------------

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
    **data
):
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

    # TODO: This index-based setting seems fragine beyond belief!  I would like to do it as
    # 'search this dict-of-dicts and replace the innder-dict that has name-key Y'
    schema["data"][0] = {"name": "binned", "values": desc}
    schema["data"][1] = {"name": "xref", "values": [{"value": v} for v in xrefs]}
    schema["data"][2] = {"name": "yref", "values": [{"count": v} for v in yrefs]}

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

def resize(schema: Dict[str, Any], *, w: int = None, h: int = None):
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
