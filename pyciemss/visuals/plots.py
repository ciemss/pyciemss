import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import IPython.display
import vl_convert

from .barycenter import triangle_contour
from .calibration import calibration
from .graphs import attributed_graph, spring_force_graph
from .histogram import heatmap_scatter, histogram_multi, map_heatmap
from .trajectories import trajectories
from .vega import VegaSchema, orient_legend, pad, rescale, resize, set_title

__all__ = [
    "VegaSchema",
    "pad",
    "resize",
    "set_title",
    "rescale",
    "orient_legend",
    "triangle_contour",
    "trajectories",
    "calibration",
    "histogram_multi",
    "attributed_graph",
    "spring_force_graph",
    "heatmap_scatter",
    "map_heatmap",
]

_output_schema = Path(__file__).parent / "modified_schema"
_output_html = Path(__file__).parent / "html"


def save_schema(schema: Dict[str, Any], path: Path):
    """Save the schema using common convention"""
    with open(path, "w") as f:
        json.dump(schema, f, indent=3)


def check_geoscale(schema):
    geoscale = False
    if "signals'" in schema.keys():
        for i in range(len(schema["signals"])):
            signal = schema["signals"][i]
            if "on" in signal.keys():
                if "geoscale" in signal["on"][0]["update"].lower():
                    geoscale = True
    return geoscale


def ipy_display(
    schema: Dict[str, Any],
    *,
    format: Literal["png", "svg", "PNG", "SVG", "interactive", "INTERACTIVE"] = "png",
    force_clear: bool = False,
    dpi: Optional[int] = None,
    **kwargs,
):
    """Wrap for dispaly in an ipython notebook.
    schema -- A vega JSON schema ready for rendering
    format -- Return a PNG or SVG if requested (wrapped for jupyter dipslay)
                OR for "interactive" returns None but displays the schema
              Format specifier is case-insensitive.
    force_clear -- Force clear the result cell (sometimes required after an error)
    dpi -- approximates DPI for output (other factors apply)
    **kwargs -- Passed on to the selected vl_convert function

    The vlc_convert PNG export function takes a 'scale' factor,
    which uses in machine-dependent resolution units.
    If dpi is specified and kwargs DOES NOT include a scale,
    then dpi will be used to estimate the scale factor to get a result near the requested
    resolution assuming a machine-resolution of 72dpi.

    This (may) return PNG data **wrappped** for display in jupyter.
    The raw PNG data is accessible from the returned objects `data` property. Save to a file as:
    ```
    image = display(schema)
    with open("test_image.png", "wb") as f:
       f.write(image.data)
    ```

    return -- PNG data to display (if not interactive) OR an interactive vega plot

    """
    if force_clear:
        IPython.display.clear_output(wait=True)

    if check_geoscale(schema):
        schema_path = _output_schema / "modified_map_heatmap.json"
        html_location = _output_html / "visualize_map.html"
        save_schema(schema, schema_path)
        print(
            f"Schema includes 'geoscale' which can't be interactively rendered. Open html file at {html_location}"
        )
    elif format in ["interactive", "INTERACTIVE"]:
        bundle = {"application/vnd.vega.v5+json": schema}
        print("", end=None)
        IPython.display.display(bundle, raw=True)

    elif format in ["png", "PNG"]:
        if dpi and "scale" not in kwargs:
            kwargs["scale"] = dpi // 72
        png_data = vl_convert.vega_to_png(schema, **kwargs)
        return IPython.display.Image(png_data)

    elif format in ["svg", "SVG"]:
        png_data = vl_convert.vega_to_svg(schema, **kwargs)
        return IPython.display.SVG(png_data)
    else:
        raise ValueError(f"Unhandled format requested: {format}")
