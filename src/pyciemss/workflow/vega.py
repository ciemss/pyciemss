from copy import deepcopy
import IPython.display
import pandas as pd
import numpy as np

import importlib.resources
import json

_resource_root = importlib.resources.files("pyciemss.workflow")
histogram_multi_schema = _resource_root.joinpath("histogram_static_bins_multi.vg.json")

def histogram_multi(*, xrefs=[],
                     yrefs=[],
                     bins=50, 
                     return_bins=False,
                   **data):
    """
    Create a histogram with server-side binning.
    
    TODO: Maybe compute overlap in bins explicitly and visualize as stacked?
          Limits (practically) to two distributions, but legend is more clear.
    
    data - Data to plot
    xrefs - List of values in the bin-range to highlight as vertical lines 
    yrefs - List of values in the count-range to highlight as horizontal lines 
    bins - Number of bins to divide into
    """
    def hist(label, subset):
        try:
            subset = subset["state_values"]
        except:
            subset = subset
            
        counts, edges = np.histogram(subset, bins=bins)
        spans = [*(zip(edges, edges[1:]))]
        desc = [{"bin0": l.item(), "bin1": h.item(), "count": c.item(), "label": label} 
                for ((l, h), c) in zip(spans, counts)]
        return desc
    
    hists = [hist(label, group) for label, group in data.items()]
    desc = [item for sublist in hists
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
    