from copy import deepcopy
import IPython.display
import pandas as pd
import numpy as np

import importlib.resources
import json

_resource_root = importlib.resources.files("pyciemss.workflow")
histogram_multi_schema = _resource_root.joinpath("histogram_static_bins_multi.vg.json")


def sturges_bin(data):
    """Determine number of bin susing sturge's rule.  
    TODO: Consider a Freedman-Diaconis rule (larger data sizes, large spread in data...)
    """
    return int(np.ceil(np.log2(len(data))) + 1)


def histogram_multi(*, xrefs=[],
                     yrefs=[], 
                     bin_rule=sturges_bin,
                     return_bins=False,
                   **data):
    """
    Create a histogram with server-side binning.

    
    TODO: Maybe compute overlap in bins explicitly and visualize as stacked?
          Limits (practically) to two distributions, but legend is more clear.
    
    TODO: Need to align histogram bin size between groups to make the visual representation more interpretable
    
    **data -- Keyword-tagged datasets.  Key will become a label.
    bin_rule -- Determines bins width using this function. Will received a joint dataframe of all data passed
    xrefs - List of values in the bin-range to highlight as vertical lines 
    yrefs - List of values in the count-range to highlight as horizontal lines 
    bins - Number of bins to divide into
    """
    def hist(label, subset, edges):
        assignments = np.digitize(subset, edges)-1
        counts = np.bincount(assignments)
        spans = [*(zip(edges, edges[1:]))]
        desc = [{"bin0": l.item(), "bin1": h.item(), "count": c.item(), "label": label} 
                for ((l, h), c) in zip(spans, counts)]
        return desc
    
    def as_value_list(label, data):
        try:
            return data["state_values"].rename(label)
        except:
            return pd.Series(data).rename(label)

    data = {k: as_value_list(k, subset) for k, subset in data.items()}

    joint = pd.concat(data)
    bins_count = sturges_bin(joint)
    counts, edges = np.histogram(joint, bins=bins_count)

    hists = {k: hist(k, subset, edges) for k, subset in data.items()}
    desc = [item for sublist in hists.values()
             for item in sublist]
    
    with open (histogram_multi_schema) as f:
        schema = json.load(f)

    #TODO: This index-based setting seems fragine beyond belief!  I would like to do it as
    # 'search this dict-of-dicts and replace the innder-dict that has name-key Y'
    schema["data"][0] = {"name": "binned", "values": desc}
    schema["data"][1] = {"name": "xref", 
                         "values": [{"value": v} for v in xrefs]}
    schema["data"][2] = {"name": "yref", 
                         "values": [{"count": v} for v in yrefs]}
    
    if return_bins:
        return schema, pd.DataFrame(desc)
    else: 
        return schema    
    
    
    
#### --- Utlity functions for working with Vega ----
    

def ipy_display(spec, *, lite=False):
    """Wrap for dispaly in an ipython notebook.
    spec -- A vega JSON schema ready for rendering
    """
    if lite:
        bundle = {'application/vnd.vegalite.v5+json': spec}
    else:
        bundle = {'application/vnd.vega.v5+json': spec}

    return IPython.display.display(bundle, raw=True)

def resize(schema, *, w=None, h=None):
    """Utility for changing the size of a schema. 
    Always returns a copy of the original schema.
    
    schema -- Schema to change
    w -- New width (optional)
    h -- New height (optional)
    """
    schema = deepcopy(schema)
    
    if h is not None: schema["height"] = h
    if w is not None: schema["width"] = w

    return schema
    