from typing import List, Callable, Optional, Tuple

from numbers import Number

import pandas as pd
import numpy as np

from . import vega


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
) -> vega.VegaSchema:
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

    schema = vega.load_schema("histogram_static_bins_multi.vg.json")

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

    schema["data"] = vega.replace_named_with(schema["data"], "binned", ["values"], desc)

    schema["data"] = vega.replace_named_with(
        schema["data"], "xref", ["values"], [{"value": v} for v in xrefs]
    )

    schema["data"] = vega.replace_named_with(
        schema["data"], "yref", ["values"], [{"count": v} for v in yrefs]
    )

    if return_bins:
        return schema, pd.DataFrame(desc).set_index(["bin0", "bin1"])
    else:
        return schema


def heatmap_scatter(
    points: pd.DataFrame,
    mesh: Optional[Tuple] = None,
    *,
    x_name: Optional[str] = None,
    y_name: Optional[str] = None,
    max_x_bins: int = 10,
    max_y_bins: int = 10
) -> vega.VegaSchema:
    """
    Render a heatmap with points overlaid.
    
    The heatmap can be IMPLICIT and derived from the points or it can EXPLICIT
    and supplied as 'mesh'. If it implicit, then a binning function is used at 
    render time that does bin-boundry 'nciing.  If specific bin-count or bin-boundries
    are needed, use an explicit mesh.  
    
    points -- Set of points to render as a scatterplot.  
    mesh -- (Optional) Tripple of x-indicies, y-indicies and values. x-indicies and y-indicies
       of the same format as numpy meshgrid.  `values` as a numpy array.
    x_name: str name of column in dataset for x axis, (defaults to first column)
    y_name: str, name of column in dataset for y axis (defaults to second column)
    max_x_bins: int = 10, maximum number bins for the x axis (ignored for explicit mesh)
    max_y_bins: int = 10, maximum number bins for the y axis (ignored for explicit mesh)
    """

    def mesh_to_heatmap(mesh_data):
        """
        **mesh_data -- input as mesh data, will be converted to grids
        adding half the difference in grid spacing to each coordinate 
        so point becomes center of a grid for heatmap
        """
        xv, yv, zz = mesh_data
        half_spacing_x = (xv[0, 1] - xv[0, 0])/2
        half_spacing_y = (yv[1, 0] - yv[0, 0])/2
        dataset = pd.DataFrame({"x_start": xv.ravel() - half_spacing_x,
                                "x_end": xv.ravel() + half_spacing_x,
                                "y_start": yv.ravel() - half_spacing_y,
                                "y_end": yv.ravel() + half_spacing_y,
                                "__count": zz.ravel()})
        return dataset.to_json(orient="records")

    json_points = points.to_json(orient='records')

    if x_name is None:
        x_name = points.columns[0]
    if y_name is None:
        y_name = points.columns[1]

    if mesh is None:
        schema = vega.load_schema("heatmap_scatter.vg.json")

        schema["data"] = vega.replace_named_with(schema["data"], "points", ["values"], json_points)

        schema["data"] = vega.replace_named_with(schema["data"], "source_0", ["values"], json_points)

        schema["signals"] = vega.replace_named_with(
                schema["signals"], "max_x_bins", ["value"], max_x_bins
            )
        schema["signals"] = vega.replace_named_with(
                schema["signals"], "max_y_bins", ["value"], max_y_bins
            )

        schema["signals"] = vega.replace_named_with(
                schema["signals"], "x_name", ["value"], x_name
            )

        schema["signals"] = vega.replace_named_with(
                schema["signals"], "y_name", ["value"], y_name
            )
    else:
        schema = vega.load_schema("mesh_scatter.vg.json")
        json_heatmap = mesh_to_heatmap(mesh)
        schema["data"] = vega.replace_named_with(schema["data"], "points", ["values"], json_points)
        schema["data"] = vega.replace_named_with(schema["data"], "mesh", ["values"], json_heatmap)

    return schema

