"""Utilities for working with Vega schemas"""

import json
import pkgutil
from copy import deepcopy
from itertools import compress
from numbers import Number
from typing import Any, Dict, List, Literal, Optional, Union
from pathlib import Path

import vl_convert
import IPython

VegaSchema = Dict[str, Any]

# TODO: Make a function decorator(s) to add common propreties.
#       Things like title, pad, width, height, legend-orientation, etc.
#       Use that decorator on the various plots (instead of adding the properties ad-hoc as needed)
#       The decorator should use the variuos utility functions defined here.


def load_schema(name: str) -> VegaSchema:
    """Load a schema.

    This a module utility to load modules stored in a package.

    Args:
        name -- name of the schema file.
    """
    data = pkgutil.get_data(__name__, f"schemas/{name}")
    if data is None:
        raise ValueError(f"Could not locate requested schema file in package: {name}")
    else:
        return json.loads(data)


def size(
    schema: VegaSchema, *, w: Optional[int] = None, h: Optional[int] = None
) -> VegaSchema:
    """Utility for changing the size of a schema.
    Always returns a copy of the original schema.

    schema -- Schema to change
    w -- New width (optional)
    h -- New height (optional)

    If w and h are not provided, returns the currently set size.
    """
    if w is None and h is None:
        return {"height": schema.get("height"), "width": schema.get("width")}

    schema = deepcopy(schema)

    if h is not None:
        schema["height"] = h
    if w is not None:
        schema["width"] = w

    return schema


def rescale(
    schema: VegaSchema,
    scale_name: str,
    scaletype: str,
    zero: Literal["auto", True, False] = "auto",
) -> VegaSchema:
    """Change a scale to a new interpolation type.

    Args:
        schema (VegaSchema): Vega schema to modify
        scale_name (str): _description_. Defaults to "yscale".
        scaletype (str): _description_. Defaults to "log".
        zero (Literal["auto", True, False]): "Auto" sets to 'true' if non-log scale.

    Returns:
        VegaSchema: Copy of the schema with an updated scale
    """
    schema = deepcopy(schema)

    if zero == "auto":
        zero = scaletype not in ["log", "symlog"]

    schema["scales"] = replace_named_with(
        schema["scales"], scale_name, ["type"], scaletype
    )

    schema["scales"] = replace_named_with(schema["scales"], scale_name, ["zero"], zero)

    return schema


def set_title(
    schema: VegaSchema,
    title: Union[str, List[str]],
    *,
    target: Literal[None, "x", "y"] = None,
):
    """
    Sets the title of a plot or axis.

    Args:
        schema (_type_): Schema to modify
        title (str): Ttile to set
        target (Literal[None, x, y], optional): Axis to set or whole plot if None (default).

    Returns:
        A new schema with the title attribute set
    """
    schema = deepcopy(schema)
    if target is None:
        if "title" not in schema:
            schema["title"] = ""

        if isinstance(schema["title"], dict):
            schema["title"]["text"] = title
        elif isinstance(title, list):
            schema["title"] = {"text": title}
        else:
            schema["title"] = title
    elif target == "x":
        axis = find_keyed(schema["axes"], "name", "x_axis")
        axis["title"] = title
    elif target == "y":
        axis = find_keyed(schema["axes"], "name", "y_axis")
        axis["title"] = title

    return schema


def pad(schema: VegaSchema, qty: Optional[Number] = None) -> VegaSchema:
    """Add padding to a schema.

    Args:
        schema (VegaSchema): Schema to update
        qty (None | Number): Value to set padding to
         If None, removes padding if present

    Returns:
        VegaSchema: Schema with modified padding
    """
    schema = deepcopy(schema)
    if qty is None:
        if "padding" in schema:
            del schema["padding"]
    else:
        schema["padding"] = qty

    return schema


def orient_legend(schema: VegaSchema, name: str, location: Optional[str]):
    schema = deepcopy(schema)
    legend = find_keyed(schema["legends"], "name", name)

    if location is None:
        if "orient" in legend:
            del legend["orient"]
    else:
        legend["orient"] = location

    return schema


# JSON-manipulation utilities...

def find_named(ls: List[dict], name: str, *, key="name"):
    """Find the thing in the list with dict key 'name' equal to the passed string"""
    return find_keyed(ls, key, name)


def find_keyed(ls: List[dict], key: str, value: Any):
    """In the list of dicts, finds a think where key=value"""
    for e in ls:
        try:
            if e[key] == value:
                return e
        except Exception:
            pass

    raise ValueError(f"Attempted to find, but {key}={value} not found.")


def delete_named(ls: List, name: str) -> List:
    "REMOVE the first thing in ls where 'name'==name"

    def _maybe_keep(e, found):
        if found[0]:
            return True

        if e.get("name", None) == name:
            found[0] = True
            return False
        return True

    found = [False]
    filter = [_maybe_keep(e, found) for e in ls]
    if not found[0]:
        raise ValueError(f"Attempted to remove, but {name=} not found.")

    return [*compress(ls, filter)]


def replace_named_with(ls: List, name: str, path: List[Any], new_value: Any) -> List:
    """Rebuilds the element with the given 'name' entry.

    An element is "named" if it has a key named "name".
    Only replaces the FIRST item so named.

    name -- Name to look for
    path -- Steps to the thing to actually replace
    new_value -- Value to place at end of path
    Path is a list of steps to take into the named entry.

    """

    def _maybe_replace(e, done_replacing):
        if done_replacing[0]:
            return e

        if "name" in e and e["name"] == name:
            e = deepcopy(e)
            part = e
            if path:
                for step in path[:-1]:
                    part = part[step]
                part[path[-1]] = new_value
            else:
                e = new_value

            done_replacing[0] = True

        return e

    done_replacing = [False]
    updated = [_maybe_replace(e, done_replacing) for e in ls]

    if not done_replacing[0]:
        raise ValueError(f"Attempted to replace, but {name=} not found.")

    return updated


def save_schema(schema: Dict[str, Any], path: Path):
    """Save the schema using common convention"""
    with open(path, "w") as f:
        json.dump(schema, f, indent=3)


def display(
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

    if format in ["interactive", "INTERACTIVE"]:
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
