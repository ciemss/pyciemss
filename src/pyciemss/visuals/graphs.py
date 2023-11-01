from typing import Union, Optional
from numbers import Number
import networkx as nx

from . import vega


def attributed_graph(
    graph: nx.Graph, *, collapse_all: bool = False, node_labels: Union[str, None] = None
) -> vega.VegaSchema:
    """Draw a graph with node/edge attributions color-coded.

    graph (networkx graph): A networkX graph with attributes on nodes/edges as follows.
       attribution -- Node attribute.  List of values used to color portions of the node border.
       attribution -- Edge attribute.  list of values used to color the edge.

       NOTE: The attribution property for nodes or edges MUST be a list (even if there is only one value)
       NOTE: Other properties will be propogated through.

       TODO: Make layout in the vega schema optional (accept passed- in schemas)
       The color coding of attribution will be shared between nodes and edges.

    node_labels
       Node labels will be constructed as follows:
        -- If node_labels is None, the node-id will be used
        -- If node_labels is not "label", the attribute specified will be copied into a "label" attribute
    collapse_all: If True AND an attribution includes all of the attributions used,
                the attribution will be set to just "*all*".  This can be used to
                simplify the visualization in some cases.
    """

    if node_labels is None:
        graph = nx.convert_node_labels_to_integers(graph, label_attribute="label")
    elif node_labels == "label":
        nx.set_node_attributes(
            graph, nx.get_node_attributes(graph, node_labels), "label"
        )

    gjson = nx.json_graph.node_link_data(graph)

    possible_attributions = set()
    for n in gjson["nodes"]:
        if "attribution" not in n or len(n["attribution"]) == 0:
            raise ValueError(
                f"Every node must have an 'attribution' property with at least one element in the list. Failed for {n}"
            )
        possible_attributions.update(n["attribution"])

    for e in gjson["links"]:
        if "attribution" not in e or len(e["attribution"]) == 0:
            raise ValueError(
                f"Every edge must have an 'attribution' property with at least one element in the list. Failed for {e}"
            )
        possible_attributions.update(n["attribution"])

    schema = vega.load_schema("multigraph.vg.json")

    if collapse_all:
        # This is category20, but with the light-gray moved up and the dark-gray removed
        colormap = [
            "#c7c7c7",
            "#1f77b4",
            "#aec7e8",
            "#ff7f0e",
            "#ffbb78",
            "#2ca02c",
            "#98df8a",
            "#d62728",
            "#ff9896",
            "#9467bd",
            "#c5b0d5",
            "#8c564b",
            "#c49c94",
            "#e377c2",
            "#f7b6d2",
            "#bcbd22",
            "#dbdb8d",
            "#17becf",
            "#9edae5",
        ]
        did_replace = False
        for n in gjson["nodes"]:
            if len(possible_attributions.difference(n["attribution"])) == 0:
                n["attribution"] = ["*all*"]
                did_replace = True

        for e in gjson["links"]:
            if len(possible_attributions.difference(e["attribution"])) == 0:
                e["attribution"] = ["*all*"]
                did_replace = True

        if did_replace:
            schema["scales"] = vega.replace_named_with(
                schema["scales"], "color", ["range"], colormap
            )

    schema["data"] = vega.replace_named_with(
        schema["data"], "node-data", ["values"], gjson["nodes"]
    )
    schema["data"] = vega.replace_named_with(
        schema["data"], "link-data", ["values"], gjson["links"]
    )

    return schema


def spring_force_graph(
    graph: nx.Graph,
    node_labels: Union[str, None] = "label",
    layout: Optional[dict[any, (Number, Number)]] = None,
    directed_graph: bool = True,
) -> vega.VegaSchema:
    """Draw a general spring-force graph

    graph -- Networkx graph to draw
    input_layout -- input of locations of nodes in graph as fx and fy items in data list of dictionaries
    labels -- If it is a string, that field name is used ('label' is the default; 'id' will give the networkx node-id).
              If it is None, no label is drawn.
    """
    graph = nx.convert_node_labels_to_integers(graph, label_attribute=node_labels)
    gjson = nx.json_graph.node_link_data(graph)
    schema = vega.load_schema("spring_graph.vg.json")

    # use -100 to signify no fixed location. values will update if node is dragged
    gjson["nodes"] = [dict(item, fixedx=-100, fixedy=-100) for item in gjson["nodes"]]

    if layout:

        def _layout_get(id):
            return dict(zip(["fx", "fy"], layout.get(id, (None, None))))

        gjson["nodes"] = [
            {**item, **_layout_get(item[node_labels])} for item in gjson["nodes"]
        ]

        schema["signals"] = vega.replace_named_with(
            schema["signals"], "layoutdata", ["value"], True
        )

    schema["data"] = vega.replace_named_with(
        schema["data"], "node-data", ["values"], gjson["nodes"]
    )

    schema["data"] = vega.replace_named_with(
        schema["data"], "link-data", ["values"], gjson["links"]
    )

    if node_labels is None:
        schema["marks"] = vega.delete_named(schema["marks"], "labels")
    else:
        labels = vega.find_named(schema["marks"], "labels")
        labels["encode"]["enter"]["text"]["field"] = f"datum.{node_labels}"

    if not directed_graph:
        schema["marks"] = vega.delete_named(schema["marks"], "arrows")

    return schema
