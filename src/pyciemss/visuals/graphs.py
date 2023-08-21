from typing import Union
import networkx as nx

from . import vega


def attributed_graph(graph: nx.Graph) -> vega.VegaSchema:
    """Draw a graph with node/edge attributions color-coded.

    graph (networkx graph): A networkX graph with attributes on nodes/edges as follows.
       label -- Node attribute used to label the nodes.  Can be anything that vega can interpret as a string.
       attribution -- Node attribute.  List of values used to color portions of the node border.
       attribution -- Edge attribute.  list of values used to color the edge.

       NOTE: The attribution property for nodes or edges MUST be a list (even if there is only one value)
       NOTE: Other properties will be propogated through.

       TODO: Make layout in the vega schema optional (accept passed- in schemas)
       TODO: Add interaction to the vega schema
       The color coding of attribution will be shared between nodes and edges.
    """
    gjson = nx.json_graph.node_link_data(graph)

    for n in gjson["nodes"]:
        if "attribution" not in n or len(n["attribution"]) == 0:
            raise ValueError(
                "Every node must have an 'attribution' property with at least one element in the list."
            )

    for e in gjson["links"]:
        if "attribution" not in n or len(n["attribution"]) == 0:
            raise ValueError(
                "Every edge must have an 'attribution' property with at least one element in the list."
            )

    schema = vega.load_schema("multigraph.vg.json")

    schema["data"] = vega.replace_named_with(
        schema["data"], "node-data", ["values"], gjson["nodes"]
    )
    schema["data"] = vega.replace_named_with(
        schema["data"], "link-data", ["values"], gjson["links"]
    )

    return schema


def spring_force_graph(
    graph: nx.Graph, node_labels: Union[str, None] = "label", directed_graph: bool = True
) -> vega.VegaSchema:
    """Draw a general spring-force graph

    graph -- Networkx graph to draw
    labels -- If it is a string, that field name is used ('label' is the default; 'id' will give the networkx node-id).
              If it is None, no label is drawn.
    """
    gjson = nx.json_graph.node_link_data(graph)

    schema = vega.load_schema("spring_graph.vg.json")

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
