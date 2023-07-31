from typing import Dict, Any

import json
import IPython.display

from .vega import VegaSchema, pad, resize, set_title, rescale, orient_legend
from .barycenter import triangle_contour
from .trajectories import trajectories
from .calibration import calibration
from .histogram import histogram_multi
from .graphs import attributed_graph, spring_force_graph


__all__ = [
    VegaSchema,
    pad,
    resize,
    set_title,
    rescale,
    orient_legend,
    triangle_contour,
    trajectories,
    calibration,
    histogram_multi,
    attributed_graph,
    spring_force_graph,
]


def save_schema(schema: Dict[str, Any], path: str):
    """Save the schema using common convention"""
    with open(path, "w") as f:
        json.dump(schema, f, indent=3)


def ipy_display(spec: Dict[str, Any], *, lite=False, force_clear=False):
    """Wrap for dispaly in an ipython notebook.
    spec -- A vega JSON schema ready for rendering
    """
    if lite:
        bundle = {"application/vnd.vegalite.v5+json": spec}
    else:
        bundle = {"application/vnd.vega.v5+json": spec}

    print("", end=None)
    if force_clear:
        IPython.display.clear_output(wait=True)

    IPython.display.display(bundle, raw=True)
