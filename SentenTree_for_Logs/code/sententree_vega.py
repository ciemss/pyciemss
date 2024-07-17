from graphviz import Digraph
import json
import vega


def graphviz_dot_LR(nxG):
    gvG = Digraph(strict=True, filename="logtree.gv", engine="dot")
    gvG.graph_attr["rankdir"] = "LR"

    for node, data in nxG.nodes(data=True):
        gvG.node(node,
                 label=node,
                 fontsize=str(data["count"]),  # TODO: Match projection to the one in the vega schema (was a linear projection from 12-15 at one time)
                 ordering="out"

                 )

    for s, t in nxG.edges:
        gvG.edge(s, t)

    gvG.graph_attr["nodesep"] = ".1"
    gvG.render(filename="logtree", format="json", engine="dot")
    with open("logtree.json") as f:
        data = json.load(f)

    def _for(e):
        bounds = e["_draw_"][1]['rect']
        return {"x": bounds[0],
                "y": bounds[1],
                "width": float(e["width"]),
                "height": float(e["height"])
                }

    layout = {e["name"]: _for(e) for e in data["objects"]}
    return layout


def detag_label(tagged_label):
    """
    Convert the node-name to a textual lable.
    Assumes the node-name is a the text+occurance-tag
    parameters:
        tagged_label: (string) label with occurrence tag
    returns: label without occurrence tag
    """
    split_str = tagged_label.split()
    for i, word in enumerate(split_str):
        head, _, _ = word.rpartition("_")
        split_str[i] = head
    return " ".join(split_str)


def complete_node_data(G, ext_pos):
    for node, data in G.nodes(data=True):
        entry = ext_pos[node]
        entry["id"] = node
        entry["count"] = data["count"]
        entry["summary"] = data.get("summary", False)
        entry["label"] = detag_label(node)

    return ext_pos


def vega_sententree(G, *, w=800, h=300):
    """
    vega_sententree_from_gv: Convience function to return a Vega Specification from a Graphviz graph

    parameters:
        G (networkx graph)

    returns: A Vega specification of a SentenTree graph
    """
    vega_spec = vega.load_schema("SentenTree.json")
    layout = graphviz_dot_LR(G)

    # Update edge and node data in spec
    vega_spec["data"][0]["values"] = [*complete_node_data(G, layout).values()]
    # vega_spec["data"][1]["values"] = [{"source": s, "target": t}
    #                                   for s, t in G.edges()]
    vega_spec["data"][1]["values"] = [{"source": s, "target": t, "weight": G.edges[s, t]["weight"]}
                                    for s, t in G.edges()]
    vega_spec = vega.size(vega_spec, w=w, h=h)
    return vega_spec


