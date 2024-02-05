"""Utilities for working with Vega schemas"""

from typing import List, Dict, Any, Literal, Optional, Union
from numbers import Number

from itertools import compress
from copy import deepcopy
import pkgutil
import json

VegaSchema = Dict[str, Any]

# TODO: Make a function decorator(s) to add common propreties.
#       Things like title, pad, width, height, legend-orientation, etc.
#       Use that decorator on the various plots (instead of adding the properties ad-hoc as needed)
#       The decorator should use the variuos utility functions defined here.


def load_schema(name: str) -> VegaSchema:
    """Load a schema.

    This a module utility for use in installed pyciemss to
    load module-stored utilities.  It IS NOT for reading arbitrary schemas
    (use the json package for that).

    Args:
        name -- name of the schema file.
    """
    data = pkgutil.get_data(__name__, f"schemas/{name}")
    if data is None:
        raise ValueError(f"Could not locate requested schema file: {name}")
    else:
        return json.loads(data)


def resize(
    schema: VegaSchema, *, w: Optional[int] = None, h: Optional[int] = None
) -> VegaSchema:
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

    schema["scales"] = replace_named_with(
        schema["scales"], scale_name, ["zero"], zero
    )

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


def orient_legend(schema: VegaSchema, name: str, location: Optional[str]):
    schema = deepcopy(schema)
    legend = find_keyed(schema["legends"], "name", name)

    if location is None:
        if "orient" in legend:
            del legend["orient"]
    else:
        legend["orient"] = location

    return schema


def replace_named_with(
    ls: List, name: str, path: List[Any], new_value: Any
) -> List:
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
