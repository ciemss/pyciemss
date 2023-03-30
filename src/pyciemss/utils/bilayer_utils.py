import networkx as nx
import petri_utils
from copy import deepcopy
from itertools import groupby
import json
import re

# Look at algebriac julia...

__all__ = ["as_petri", "draw", "load"]


def as_petri(G):
    """ Converts a bilayer network to a petrinet representation.
    """

    def root_id(id):
        return id[:-1]

    G2 = nx.MultiDiGraph()
    transitions = [n for n, d in bilayerG.nodes(data=True) if d["layer"] == "mediator"]

    states = [
        *set(
            [n for n, d in bilayerG.nodes(data=True) if d["layer"] == "input"]
            + [
                root_id(n)
                for n, d in bilayerG.nodes(data=True)
                if d["layer"] == "output"
            ]
        )
    ]

    inputs = [
        {"is": states.index(a) + 1, "it": transitions.index(b) + 1}
        for a, b, d in bilayerG.edges(data=True)
        if d["type"] == "input"
    ]

    outputs = [
        {"ot": transitions.index(a) + 1, "os": states.index(root_id(b)) + 1}
        for a, b, d in bilayerG.edges(data=True)
        if d["type"] == "outcome"
    ]

    transitions = [{"tname": n} for n in transitions]
    states = [{"sname": s} for s in states]

    pG = petri_utils.load({"S": states, "T": transitions, "I": inputs, "O": outputs})

    return pG


def draw(G, subset_key="layer", ax=None):
    pos = nx.multipartite_layout(G, subset_key=subset_key)

    edge_cmap = {
        "input": "darkred",
        "recursive": "darkred",
        "outcome": "CornflowerBlue",
    }

    edge_colors = [edge_cmap[t] for t in nx.get_edge_attributes(G, "type").values()]
    nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color=edge_colors)

    node_cmap = {"input": "slategray", "output": "green", "mediator": "mediumorchid"}

    node_colors = [node_cmap[n] for n in nx.get_node_attributes(G, "layer").values()]

    nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color=node_colors)

    nx.draw_networkx_labels(G, pos=pos, ax=ax)


def load(bilayer_file):
    """Loads a path-like object to networkx representation of a bilayer network.
    """
    with open(bilayer_file) as f:
        bilayer = json.load(f)

    parameters = bilayer["parameters"]

    def find_parameter_for(law):
        for id in parameters:
            law_parts = re.split("\+|\*", law)
            if id in law_parts:
                return id

    G = nx.DiGraph()
    for template in bilayer["templates"]:
        template = deepcopy(template)
        subject = template.pop("subject")
        implicit_outcome = deepcopy(subject)
        outcome = template.pop("outcome")
        implicit_subject = deepcopy(outcome)

        template["id"] = find_parameter_for(template["rate_law"])
        template["layer"] = "mediator"
        template = {**template, **parameters[template["id"]]}

        subject["id"] = subject["name"]
        subject["layer"] = "input"

        implicit_outcome["id"] = f"{subject['name']}'"
        implicit_outcome["layer"] = "output"

        outcome["id"] = f"{outcome['name']}'"
        outcome["layer"] = "output"

        implicit_subject["id"] = implicit_subject["name"]
        implicit_subject["layer"] = "input"

        G.add_node(subject["id"], **subject)
        G.add_node(implicit_outcome["id"], **implicit_outcome)
        G.add_node(outcome["id"], **outcome)
        G.add_node(implicit_subject["id"], **implicit_subject)

        if "controller" in template:
            controller = template.pop("controller")
            controller["id"] = controller["name"]
            controller["layer"] = "input"
            G.add_node(controller["id"], **controller)

        G.add_node(template["id"], **template)

        G.add_edge(subject["id"], template["id"], type="input")
        G.add_edge(template["id"], implicit_outcome["id"], type="recursive")
        G.add_edge(template["id"], outcome["id"], type="outcome")

        if controller is not None:
            G.add_node(controller["id"], **controller)
            G.add_edge(controller["id"], template["id"], type="input")

    return G
