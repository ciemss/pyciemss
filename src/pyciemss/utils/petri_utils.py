import json
import functools
import collections
import numbers
from typing import TypedDict, Literal, TypeVar, Optional, Callable, List, Tuple, Dict, Union, NamedTuple, TYPE_CHECKING

import torch
import numpy
import pyro
import pandas as pd
import networkx as nx
from itertools import groupby
import urllib.request

if TYPE_CHECKING:
    import mira.metamodel

__all__ = ['seq_id_suffix',
           'load_sim_result',
           'load',
           'encode',
           'load_mira',
           'draw_petri',
           'natural_order',
           'add_state_indicies',
           'register_template',
           'natural_conversion',
           'natural_degradation',
           'natural_order',
           'controlled_conversion',
           'grouped_controlled_conversion',
           'deterministic',
           'petri_to_ode',
           'order_state',
           'unorder_state',
           'duplicate_petri_net',
           'intervene_petri_net',
           'get_mira_initial_values',
           'get_mira_parameter_values',
           'get_mira_parameter_names',
           'get_mira_parameters',
           'set_mira_initial_values',
           'set_mira_parameters'
]


def seq_id_suffix(df):
    """Utility that turns non-unique-names in to unique names.  (Suitable for a groupby/apply)
    """
    seen = {}
    def maybe_extend(entry):
        count = seen.get(entry, -1)
        count = count + 1
        seen[entry] = count
        if count > 0:
            entry = f"{entry}_{count}"

        return entry

    field = df.columns[0]
    return df.assign(**{"index": [maybe_extend(e) for e in df[field]]}).set_index("index")


def load_sim_result(results_file, petrisource, orient="wide")  -> pd.DataFrame:
    """Create a dataframe froma petrinet JSON definition and a results file.

    ASSUMES the results file came from a simuliation using the petri_file
    (the only check is that the number of states in the petri_file
     matches the number of measurements at each time-point.)

    Args:
        results_file (path-like): JSON file with 'states' and 'time' entries (similar to json2mtk output)
        petrisource (petrinet graph, file or url): A petrinet, similar to the one loaded by 'load'

    Returns:
        pandas dataframe: Wide-format data frame of simuliation results, column labels from the petri_file
    """
    if isinstance(petrisource, str):
        petrinet = load(url=petrisource)
    else:
        petrinet = petrisource

    if orient not in ["wide", "long"]:
        raise ValueError(f"Invalid 'orient' passed {orient}. Only 'wide' and 'narrow' accepted.")


    with open(results_file) as f:
        results = json.load(f)

    state_names = nx.get_node_attributes(petrinet, "sname").values()

    wide =  pd.DataFrame(results["states"], columns=state_names,
                         index = pd.Series(results["time"], name="time"))

    if orient == "narrow":
        pass

    elif orient == "wide":
        df = wide

    return df


def load_mira(template_model: "mira.metamodel.TemplateModel") -> nx.MultiDiGraph:
    """Generate a NetworkX output from a MIRA metamodel template specification."""
    from mira.modeling import Model
    from mira.modeling.petri import PetriNetModel

    model = Model(template_model)
    petri_net_model = PetriNetModel(model)
    petri_net_json = petri_net_model.to_json()
    return load(petri_net_json)

def load(petrisource=None, uniquer=seq_id_suffix) -> nx.MultiDiGraph:
    """ Create a newtworkx output from a petri-net specification

    Args:
        petri (optional): Dictionary-description of a petrinet (e.g., results of a json.load from a file)
               In particular, expects keys S (states), T (transitions), I (inputs), O (outputs).
               S and T are lists describing nodes,
               I and O are lists of pairs of indicies describing edges

        uniquer (optional): Function to making unique identifiers
        file (optional): path-like to load a petrinet from.
                         Ignored if petri is not None.
        url (optional): url to load a petrinet from.
                        Ignored if petri is not None or file is not None.

    Returns:
        networkx DiGraph: Digraph representing the petrinet
    """
    if isinstance(petrisource, str) and petrisource.startswith("http"):
        with urllib.request.urlopen(petrisource) as url:
            petri = json.load(url)
    elif isinstance(petrisource, str):
        with open(petrisource) as f:
            petri = json.load(f)
    else:
        ##Assumes its dictionary-like!
        petri = petrisource

    states = uniquer(pd.DataFrame(petri["S"]))
    transitions = uniquer(pd.DataFrame(petri["T"]))
    inputs = pd.DataFrame(petri["I"])
    outputs = pd.DataFrame(petri["O"])

    input_edges = [*zip(inputs['is'].map(lambda v: states.index[v-1]),
                    inputs['it'].map(lambda v: transitions.index[v-1]))]

    output_edges = [*zip(outputs['ot'].map(lambda v: transitions.index[v-1]),
                    outputs['os'].map(lambda v: states.index[v-1]))]

    G = nx.MultiDiGraph()
    st = states.assign(type="state").to_dict(orient="index")
    tr = transitions.assign(type="transition").to_dict(orient="index")
    G.add_nodes_from(st.items())
    G.add_nodes_from(tr.items())
    G.add_edges_from(input_edges + output_edges)

    return G





def draw_petri(G: nx.MultiDiGraph, ax=None) -> None:
    """Convenience utility to draw a networkx graph, assuming it represents a petrinet.

    Assumes the graph has a "type" attribute and that the type are "state" and "transition" (case sensitive).

    TODO: Does not proerly handle multiple links between the same nodes (will overplot right now).
    TODO: Improve so it accounts for the size of the text involved.
    """
    G2 = nx.MultiDiGraph()
    G2.add_nodes_from(G.nodes(data=False))
    G2.add_edges_from(G.edges)
    G2 = nx.convert_node_labels_to_integers(G2, ordering="default")
    mapping = dict(zip(G2.nodes, sorted(G.nodes)))

    pos = nx.nx_pydot.graphviz_layout(G2)
    pos = {mapping[int(n)]: values for n, values in pos.items()}

    _k = lambda v: v[1]
    types = sorted(nx.get_node_attributes(G, "type").items(), key=_k)
    types = {t: [v[0] for v in nodes]
             for t,nodes
             in groupby(types, _k)}

    def _get_id(n, d):
        try: return  d["sname"]
        except KeyError: pass

        try: return d["tname"]
        except KeyError: pass

        return n

    labels = {n:_get_id(n, d) for n, d in G.nodes(data=True)}
    print(types)
    nx.draw_networkx_edges(G, pos=pos, ax=ax)
    nx.draw_networkx_nodes(G, pos=pos, nodelist=types["state"], node_color="lightblue", node_shape="o", ax=ax)
    nx.draw_networkx_nodes(G, pos=pos, nodelist=types["transition"], node_color="darkorange", node_shape="s", ax=ax)
    nx.draw_networkx_labels(G, pos=pos, labels=labels, ax=ax)



def encode(petrinet):
    """Encode petrinet as a dictionary with the following keys:
    S -- List of state nodes (with attributes)
    T -- List of transition nodes (with attributes)
    I -- Edges from state to transition (essentially pairs of indicies in S & T)
    O -- Edges from transition to state (essentially pairs of indicies in T & S)

    returns dictionary encoding of graph, suitable for dumping to a file as JSON.
    It is intended that a round-trip encoding yields an isomorphic graph.  In code:
       `import networkx.algorithms.isomorphism as iso
       G1 = load(<file>)
       G2 = load(encode(G))
       atts = {*petri_G.nodes["S"].keys()} | {*petri_G.nodes["t1"].keys()}
       defaults = [None for _ in atts]
       assert(nx.is_isomorphic(petri_G, petri_G2, iso.categorical_node_match(atts, defaults)))
       `
    """

    # Node lists------
    def _clean(d, remove=["type"]):
        d = d.copy()
        for k in remove:
            del d[k]
        return d

    _k = lambda v: v[1]
    types = sorted(nx.get_node_attributes(petrinet, "type").items(), key=_k)
    types = {t: [v[0] for v in nodes]
             for t,nodes
             in groupby(types, _k)}

    S = [_clean(petrinet.nodes[n]) for n in types["state"]]
    T = [_clean(petrinet.nodes[n]) for n in types["transition"]]


    # Edge lists ------
    s_order = [e["sname"] for e in S]
    t_order = [e["tname"] for e in T]
    print(t_order)
    def _build_pairs(a,b):
        if a in s_order:
            return {"is": s_order.index(a)+1,
                    "it": t_order.index(b)+1
                   }
        else:
            return {"os": s_order.index(b)+1,
                    "ot": t_order.index(a)+1
                   }

    pairs = [_build_pairs(a, b) for a, b in petrinet.edges()]
    _k = lambda v: "I" if "is" in v.keys() else "O"
    edge_types = sorted(pairs, key=_k)
    edge_types = {t:[*edges]
                     for t, edges
                     in groupby(edge_types, _k)}
    json = {"S": S,
            "T": T,
            "I": edge_types["I"],
            "O": edge_types["O"]
           }
    return json


T = TypeVar("T", bound=Union[numbers.Number, torch.Tensor])
Template = Callable[[Dict[str, T], T, Tuple[T, ...]], Tuple[T, ...]]
_TEMPLATES: Dict[str, Template] = {}

def natural_order(G):
    """Assigns state indicies based on iteration order of G"""
    states = [n for n, data in G.nodes(data=True)
              if data["type"]=="state"]
    return {n: i for i, n in enumerate(states)}


def add_state_indicies(G, ordering_principle=natural_order, overwrite=False, in_place=False):
    """
    Add a 'state_index' field to G if it does not already exist.

    """
    if not in_place:
        G = G.copy()

    if not overwrite and len(nx.get_node_attributes(G, "state_index")) > 0:
        return G

    state_idx = ordering_principle(G)
    nx.set_node_attributes(G, state_idx, name="state_index")
    return G

def register_template(name: str, func: Optional[Template[T]] = None):
    if func is None:
        return functools.partial(register_template, name)

    _TEMPLATES[name] = func
    return func


@register_template("NaturalDegradation")
def natural_degradation(params: Dict[str, T], states: Tuple[T, ...], t: T) -> Tuple[T, ...]:
    raise NotImplementedError  # TODO


@register_template("NaturalConversion")
def natural_conversion(params: Dict[str, T], t: T, states: Tuple[T, ...]) -> Tuple[T, ...]:
    # e.g. (I, R) -> (I - 1, R + 1)
    rate = params["gamma"]
    return -rate * states[0], rate * states[0]


@register_template("ControlledConversion")
def controlled_conversion(params: Dict[str, T], t: T, states: Tuple[T, ...]) -> Tuple[T, ...]:
    # e.g. (S, I) -> (S - 1, I + 1)
    rate = params["beta"]
    return -rate * states[0] * states[1], rate * states[0] * states[1]


@register_template("GroupedControlledConversion")
def grouped_controlled_conversion(params: Dict[str, T], t: T, states: Tuple[T, ...]) -> Tuple[T, ...]:
    raise NotImplementedError  # TODO


@torch.fx.wrap
def deterministic(name, value, *, event_dim=None):
    return pyro.deterministic(name, value, event_dim=event_dim)


def petri_to_ode(
    G: nx.MultiDiGraph,
    funcs: Optional[Dict[str, Callable[[Tuple[T, ...], T], Tuple[T, ...]]]] = None,
) -> Callable[[T, Tuple[T, ...]], Tuple[T, ...]]:
    """Create an ODE system from a petri-net definition.

    Args:
        G (nx.MultiDiGraph): Petri-net graph

    Returns
        Callable: Function that takes a list of state values and returns a list of derivatives.
    """
    state2ind = {node: data["state_index"] for node, data in G.nodes(data=True)
                 if data["type"] == "state"}

    if funcs is None:
        funcs = {}

    for node, data in G.nodes(data=True):
        if data["type"] == "transition" and node not in funcs:
            funcs[node] = functools.partial(
                _TEMPLATES[data["template_type"]],
                {data['parameter_name']: data['parameter_value']}
            )

    def dX_dt(t: T, state: Tuple[T, ...]) -> Tuple[T, ...]:
        states = unorder_state(G, *(state[i] for i in range(len(state2ind))))
        derivs = {k: 0. for k in state2ind}
        for node, data in G.nodes(data=True):
            if data["type"] == "transition":
                node_input_names = {e[0] for e in G.in_edges(node, data=True)} | \
                    {e[1] for e in G.out_edges(node, data=True)}
                node_inputs = order_state(G, **{k: states[k] for k in node_input_names})
                du_dts = funcs[node](t, node_inputs)
                for i, name in enumerate(sorted(node_input_names, key=lambda k: state2ind[k])):
                    derivs[name] += deterministic(f"partial_{node}_{name}", du_dts[i])

        return order_state(G, **{name: deterministic(f"d{name}_dt", dx_dt) for name, dx_dt in derivs.items()})

    return dX_dt


def order_state(G: nx.MultiDiGraph, **states: T) -> Tuple[T, ...]:
    if not states:
        raise ValueError
    state2ind = {node: data["state_index"] for node, data in G.nodes(data=True)
                 if data["type"] == "state"}
    return tuple(states[name] for name in sorted(states.keys(), key=lambda k: state2ind[k]))


def unorder_state(G: nx.MultiDiGraph, *states: T) -> Dict[str, T]:
    ind2state = {data["state_index"]: node for node, data in G.nodes(data=True)
                 if data["type"] == "state"}
    return {ind2state[ind]: states[ind] for ind in ind2state}


def duplicate_petri_net(G: nx.MultiDiGraph, *, suffix: str = "cf") -> nx.MultiDiGraph:
    """Make a petrinet with *duplicate* intervention nodes.
    The resulting graph has all of the nodes/edges/attributes of the passed petrinet
    and a new set of nodes/edges/attributes that duplicate the originals BUT have a suffix added.

    NOTE: This method will DUPLICATE the unique IDs added by `add_state_indicies`, so that
    function *may* need to be re-run.

    Args:
        G (nx.MultiDiGraph): Petri-net graph

    Returns
        nx.MultiDiGraph: New graph with interventions applied
    """
    G2 = nx.MultiDiGraph()
    G2.add_nodes_from(G.nodes(data=True))
    G2.add_edges_from(G.edges(data=True))
    G2.add_nodes_from(
        (f"{node}_{suffix}", data) for node, data in G.nodes(data=True)
    )
    G2.add_edges_from(
        (f"{u}_{suffix}", f"{v}_{suffix}", data) for u, v, data in G.edges(data=True)
    )
    return G2


def intervene_petri_net(
    G: nx.MultiDiGraph,
    **interventions: Optional[nx.MultiDiGraph],
) -> nx.MultiDiGraph:
    """Intervene on a petri-net graph.

    Args:
        G (nx.MultiDiGraph): Petri-net graph
        interventions (Optional[nx.MultiDiGraph]): Interventions to apply

    Returns
        nx.MultiDiGraph: New graph with interventions applied
    """
    G2 = G.copy()

    for name, fragment in interventions.items():
        if fragment is None:
            assert name in G.nodes, f"Cannot remove nonexistent transition {name}"
        else:
            assert set(
                v for v, data in fragment.nodes(data=True)
                if data["type"] == "transition"
            ) == {name}, f"Intervention {name} must contain exactly one transition"

        if name in G.nodes:
            G2.remove_node(name)  # XXX does this remove edges too?

        # Add nodes and edges from the intervention graph fragment.
        for node, data in fragment.nodes(data=True):
            if node not in G2.nodes:
                G2.add_node(node, **data)

        for u, v, data in fragment.edges(data=True):
            G2.add_edge(u, v, **data)

    return G2


def get_mira_initial_values(petri_net):
    return {
        node: data['mira_initial_value']
        for node, data in petri_net.nodes(data=True)
        if 'mira_initial_value' in data
    }

def get_mira_parameter_values(petri_net):
    return {
        data['parameter_name']: data['parameter_value']
        for node, data in petri_net.nodes(data=True)
        if 'parameter_name' in data and 'parameter_value' in data
    }

def get_mira_parameter_names(petri_net):
    return {
        node: data['parameter_name']
               for node, data in petri_net.nodes(data=True)
               if 'parameter_name' in data
        }

def set_mira_initial_values( petri_net, initial_values ):
    for node, data in petri_net.nodes(data=True):
        if node in initial_values:
            data['mira_initial_value'] = initial_values[node]
    return petri_net


def set_mira_parameters( petri_net, parameter_values ):
    for node, data in petri_net.nodes(data=True):
        if node in parameter_values:
            data['parameter_name'] = parameter_values[node]['parameter_name']
            data['parameter_value'] = parameter_values[node]['parameter_value']
    return petri_net

def get_mira_parameters( petri_net ):
    return {
        node: {data['parameter_name']: data['parameter_value']}
        for node, data in petri_net.nodes(data=True)
        if 'parameter_name' in data and 'parameter_value' in data
        }
