from typing import List, Callable

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
    data: pd.DataFrame,
    x_name: str = "x_name",
    y_name: str = "y_name",
    x_bin: int = 10,
    y_bin: int = 10
) -> vega.VegaSchema:
    """
    **data -- Datasets as pandas dataframe, should contain x_name and y_name,
    x_name: str name of column in dataset for x axis,
    y_name: str, name of column in dataset for y axis,
    x_bin: int = 10, max bins by x axis,
    y_bin: int = 10, max bins by y axis,
    """
    json_dict = data.to_json(orient = 'records')
    schema = vega.load_schema("heatmap_scatter.vg.json")

    schema["data"] = vega.replace_named_with(schema["data"], "points", ["values"], json_dict)

    schema["data"] = vega.replace_named_with(schema["data"], "source_0", ["values"], json_dict)

    schema["signals"] = vega.replace_named_with(
            schema["signals"], "bandwidthX", ["value"], x_bin
        )
    schema["signals"] = vega.replace_named_with(
            schema["signals"], "bandwidthY", ["value"], y_bin
        )
    schema["signals"] = vega.replace_named_with(
            schema["signals"], "x_name", ["value"], x_name
        )

    schema["signals"] = vega.replace_named_with(
            schema["signals"], "y_name", ["value"], y_name
        )

    return schema



def mesh_scatter(
    mesh_data,
    scatter_data: pd.DataFrame,
) -> vega.VegaSchema:
    """
    **mesh_data -- input as mesh data, will be converted to grids
    **scatter_data -- with alpha and gamma data
    """

    schema = vega.load_schema("mesh_scatter.vg.json")

    def mesh_to_heatmap(mesh_data):
        """
        **mesh_data -- input as mesh data, will be converted to grids
        adding half the difference in grid spacing to each coordinate 
        so point becomes center of a grid for heatmap
        """
        xv, yv, zz = mesh_data
        half_spacing_x = (xv[0, 1] - xv[0, 0])/2
        half_spacing_y = (yv[1, 0] - yv[0, 0])/2
        dataset = pd.DataFrame({"x_start": xv.ravel() - half_spacing_x, \
                                "x_end": xv.ravel() + half_spacing_x, \
                                    "y_start": yv.ravel() - half_spacing_y,     
                                    "y_end": yv.ravel() + half_spacing_y,
                                    '__count': zz.ravel()})
        return dataset.to_json(orient="records")
        
    # convert to json
    json_heatmap = mesh_to_heatmap(mesh_data)
    json_scatter = scatter_data.to_json(orient="records")

    # update data in schema
    schema["data"] = vega.replace_named_with(schema["data"], "points", ["values"], json_scatter)
    schema["data"] = vega.replace_named_with(schema["data"], "mesh", ["values"], json_heatmap)

    return schema