from typing import Optional
import pandas as pd
import numpy as np
import torch

import matplotlib.tri as tri
from pyro.distributions import Dirichlet

from . import vega


def triangle_weights(samples, concentration=20, subdiv=7):
    # Adapted from https://blog.bogatron.net/blog/2014/02/02/visualizing-dirichlet-distributions/
    # TODO: This method works...but it quite the monstrosity!  Look into ways to simplify...

    AREA = 0.5 * 1 * 0.75**0.5

    def _tri_area(xy, pair):
        return 0.5 * np.linalg.norm(np.cross(*(pair - xy)))

    def _xy2bc(xy, tol=1.0e-4):
        """Converts 2D Cartesian coordinates to barycentric."""
        coords = np.array([_tri_area(xy, p) for p in pairs]) / AREA
        return np.clip(coords, tol, 1.0 - tol)

    corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)

    # For each corner of the triangle, the pair of other corners
    pairs = [corners[np.roll(range(3), -i)[1:]] for i in range(3)]
    # The area of the triangle formed by point xy and another pair or points

    # convert to coordinates with 3, rather than to points of reference for Direichlet input
    points = torch.tensor(np.array([(_xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]))
    points /= torch.sum(points, dim=1, keepdim=True)

    alpha = samples * concentration
    vals = torch.stack(
        [
            torch.exp(Dirichlet(alpha).log_prob(points[i, :]))
            for i in range(points.shape[0])
        ]
    )
    vals /= torch.max(vals, dim=0, keepdim=True)[0]
    vals = torch.sum(vals, dim=1)
    vals /= torch.sum(vals)

    coordinates_dict = {}

    # skip every line as alternates half of each lines
    y_num = 0
    not_use_trimesh_y = []
    for y in np.unique(trimesh.y):
        y_num += 1
        if y_num % 2 == 0:
            not_use_trimesh_y.append(y)

    df_coord = pd.DataFrame({"x": trimesh.x, "y": trimesh.y, "z": vals.tolist()})
    not_use_trimesh_x = list(
        np.unique(df_coord[df_coord.y == not_use_trimesh_y[0]]["x"].tolist())
    )

    # save all existing coordinates
    for x, y, z in zip(trimesh.x, trimesh.y, vals):
        coordinates_dict[(x, y)] = z.item()

    # fill in missing part of square grid
    for x in np.unique(trimesh.x):
        for y in np.unique(trimesh.y):
            if (x, y) not in coordinates_dict.keys():
                coordinates_dict[x, y] = 0

    # convert to dataframe and sort with y first in descending order
    df = pd.DataFrame(coordinates_dict.items(), columns=["x,y", "val"])
    df[["x", "y"]] = pd.DataFrame(df["x,y"].tolist(), index=df.index)
    df = df.sort_values(["y", "x"], ascending=[False, True])

    # remove the alternative values, (every other y and all the values associated with that y)
    df_use = df[(~df.x.isin(not_use_trimesh_x)) & (~df.y.isin(not_use_trimesh_y))]

    json_dict = {}
    json_dict["width"] = len(np.unique(df_use.x))
    json_dict["height"] = len(np.unique(df_use.y))
    json_dict["values"] = df_use["val"].tolist()

    return json_dict


def triangle_contour(
    data: pd.DataFrame, *, title: Optional[str] = None, contour: bool = True
) -> vega.VegaSchema:
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
    mesh_data = triangle_weights(data)

    schema = vega.load_schema("barycenter_triangle.vg.json")
    schema["data"] = vega.replace_named_with(
        schema["data"],
        "contributions",
        ["values"],
        mesh_data,
    )

    if title:
        schema = vega.set_title(schema, title)

    if not contour:
        contours = vega.find_keyed(schema["marks"], "name", "_contours")
        contours["encode"]["enter"]["stroke"] = {
            "scale": "color",
            "field": "contour.value",
        }

    return schema
