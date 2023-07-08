from copy import deepcopy
import IPython.display
import pandas as pd
import numpy as np
import json

def ipy_display(spec, *, lite=False):
    """Wrap for dispaly in an ipython notebook
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

def regression(datasource, **kwargs):
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
    with open("regression.vg.json") as f:
        schema = json.load(f)
  
    if isinstance(datasource, pd.DataFrame):
        del schema["data"][0]["url"]
            
        schema["data"][0]["values"] = datasource.to_dict(orient="records")
    else:
        schema["data"][0]["url"] = datasource
        schema["data"][0]["format"] = kwargs
        
    return schema


def heatmap(datasource, **kwargs):
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
    with open("heatmap_cube.vg.json") as f:
        schema = json.load(f)
  
    if isinstance(datasource, pd.DataFrame):
        del schema["data"][0]["url"]
            
        schema["data"][0]["values"] = datasource.to_dict(orient="records")
    else:
        schema["data"][0]["url"] = datasource
        schema["data"][0]["format"] = kwargs
        
    return schema


def ridge(datasource, **kwargs):
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
    with open("ridges.vg.json") as f:
        schema = json.load(f)
  
    if isinstance(datasource, pd.DataFrame):
        del schema["data"][0]["url"]
            
        schema["data"][0]["values"] = datasource.to_dict(orient="records")
    else:
        schema["data"][0]["url"] = datasource
        schema["data"][0]["format"] = kwargs
        
    return schema


def violin(datasource, **kwargs):
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
    with open("violin_plot.vg.json") as f:
        schema = json.load(f)
  
    if isinstance(datasource, pd.DataFrame):
        del schema["data"][0]["url"]
            
        schema["data"][0]["values"] = datasource.to_dict(orient="records")
    else:
        schema["data"][0]["url"] = datasource
        schema["data"][0]["format"] = kwargs
        
    return schema


def contour(datasource, *, origin=None, **kwargs):
    """Create a contour plot from the passed datasource.
    
    datasource -- 
      * filename: File to load data from that will be loaded via vega's "url" facility
                  
                  Path should be relative to the runnign file-server, as they will be 
                  resolved in that context. If in a notebook, it is relative to the notebook
                  (not the root notebook server processes).
      * dataframe: A dataframe ready for rendering.  The data will be inserted into the schema 
                as a record-oriented dictionary.  The "Origin" column is used to label data.

    origin -- If passing a dataframe, what label to put on contours.
    kwargs -- If passing filename, extra parameters to the vega's url facility


    TODO: Make it so if 'origin' is not supplied, it will use the first column of the data frame.
          Might be done with soemthing like glob or jsonpatch.  Its meta-program-leve so it 
          can't be done with a signal.
    """
    with open("contour_plot.vg.json") as f:
        schema = json.load(f)
  
    if isinstance(datasource, pd.DataFrame):
        del schema["data"][0]["url"]
        
        if origin is not None:
            datasource.assign(Origin=origin)
            
        schema["data"][0]["values"] = datasource.to_dict(orient="records")
    else:
        schema["data"][0]["url"] = datasource
        schema["data"][0]["format"] = kwargs
        
    return schema


def histogram_static(data, *, 
                     xref=[],
                     yref=[],
                     bins=50, 
                     return_bins=False):
    """
    Create a histogram with server-side binning.
    
    data - Data to plot
    xref - List of values in the bin-range to highlight as vertical lines 
    yref - List of values in the count-range to highlight as horizontal lines 
    bins - Number of bins to divide into
    """
    subset = data["state_values"]
    counts, edges = np.histogram(subset, bins=bins)
    spans = [*(zip(edges, edges[1:]))]
    desc = [{"bin0": l.item(), "bin1": h.item(), "count": c.item()} 
            for ((l, h), c) in zip(spans, counts)]
    
    with open ("histogram_static_bins.vg.json") as f:
        schema = json.load(f)

    #TODO: This index-based setting seems fragine beyond belief!  I would like to do it as
    # 'search this dict-of-dicts and replace the innder-dict that has name-key Y'
    schema["data"][0] = {"name": "binned", "values": desc}
    schema["data"][1] = {"name": "xref", 
                         "values": [{"value": v} for v in xref]}
    schema["data"][2] = {"name": "yref", 
                         "values": [{"count": v} for v in yref]}
    
    if return_bins:
        return pd.DataFrame(desc), schema
    else: 
        return schema
    
    
def histogram_multi(*, xref=[],
                     yref=[],
                     bins=50, 
                     return_bins=False,
                   **data):
    """
    Create a histogram with server-side binning.
    
    data - Data to plot
    xref - List of values in the bin-range to highlight as vertical lines 
    yref - List of values in the count-range to highlight as horizontal lines 
    bins - Number of bins to divide into
    """
    def hist(label, subset):
        subset = subset["state_values"]
        counts, edges = np.histogram(subset, bins=bins)
        spans = [*(zip(edges, edges[1:]))]
        desc = [{"bin0": l.item(), "bin1": h.item(), "count": c.item(), "label": label} 
                for ((l, h), c) in zip(spans, counts)]
        return desc
    
    hists = [hist(label, group) for label, group in data.items()]
    desc = [item for sublist in hists
             for item in sublist]
    
    with open ("histogram_static_bins_multi.vg.json") as f:
        schema = json.load(f)

    #TODO: This index-based setting seems fragine beyond belief!  I would like to do it as
    # 'search this dict-of-dicts and replace the innder-dict that has name-key Y'
    schema["data"][0] = {"name": "binned", "values": desc}
    schema["data"][1] = {"name": "xref", 
                         "values": [{"value": v} for v in xref]}
    schema["data"][2] = {"name": "yref", 
                         "values": [{"count": v} for v in yref]}
    
    if return_bins:
        return pd.DataFrame(desc), schema
    else: 
        return schema    
    
def calibrate_vega(datasource, **kwargs):
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
    with open("calibrate_chart.vg.json") as f:
        schema = json.load(f)
  
    if isinstance(datasource, pd.DataFrame):
        del schema["data"][0]["url"]
            
        schema["data"][0]["values"] = datasource.to_dict(orient="records")
    else:
        schema["data"][0]["url"] = datasource
        schema["data"][0]["format"] = kwargs
        
    return schema

def triangle_contour_vega(datasource, **kwargs):
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
    with open("contour.vg.json") as f:
        schema = json.load(f)
  
    if isinstance(datasource, pd.DataFrame):
        del schema["data"][0]["url"]
        schema["data"][0]["values"] = datasource.to_dict(orient="records")
    elif isinstance(datasource, str):
        del schema["data"][0]["url"]
        schema["data"][0]["values"] =datasource
    else:
        schema["data"][0]["url"] = datasource
        schema["data"][0]["format"] = kwargs
        
    return schema




def polygon_contour_vega(datasource, **kwargs):
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
    with open("/Users/oost464/Library/CloudStorage/OneDrive-PNNL/Desktop/projects/pyciemss/notebook/integration_demo/results_petri/contour.vg.json") as f:
        schema = json.load(f)
  
    if isinstance(datasource, pd.DataFrame):
        schema["data"][0]["values"] = datasource.to_dict(orient="records")
    
    elif isinstance(datasource, dict):
        schema["data"][0]["values"] = {'width': datasource['width'], 'height': datasource['height'], 'values': datasource['values']}
        schema["data"][2]["values"] = datasource
        schema["width"] = datasource['width']
        schema["height"] = datasource['height']
        
    return schema