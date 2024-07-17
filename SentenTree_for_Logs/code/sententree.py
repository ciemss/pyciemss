from collections import Counter

import networkx as nx
from prefixspan import PrefixSpan

import patterns


def tag_numbers_with_index_word_with_occurance(sents):
    """
    Tags numbes by their index in the sentence.
    Tags words by their occurance count.

    parameters:
        sents: list of lists, where a sublist represents a sentence

    returns: a list of lists, where elements of a sublist are tagged by their occurrence
    """
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    for sent in sents:
        words_seen = {}
        for i, word in enumerate(sent):
            if is_number(word):
                sent[i] = f"{word}_{i}"
            else:
                words_seen[word] = words_seen.get(word, 1)
                sent[i] = f"{word}_{words_seen[word]}"

    return sents


def tag_numbers_with_words_words_with_occurance(sents):
    """
    Tags numbes by the word that preceeds them.
    Tags words by their occurance count.

    parameters:
        sents: list of lists, where a sublist represents a sentence

    returns: a list of lists, where elements of a sublist are tagged by their occurrence
    """
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _detag(word):
        head, _, _ = word.rpartition("_")
        return head

    for sent in sents:
        words_seen = {}
        for i, word in enumerate(sent):
            if is_number(word):
                tag = _detag(sent[i-1]) if i > 0 else "_start"
                sent[i] = f"{word}_{tag}"
            else:
                words_seen[word] = words_seen.get(word, 1)
                sent[i] = f"{word}_{words_seen[word]}"

    return sents


def tag_words_with_index(sents):
    """
    tag_words: adds the occurrence to a word for each word in a sublist

    parameters:
        sents: list of lists, where a sublist represents a sentence

    returns: a list of lists, where elements of a sublist are tagged by their occurrence
    """
    for sent in sents:
        for i, word in enumerate(sent):
            sent[i] = word + "_" + str(i)
    return sents


def tag_words_with_occurrence(sents):
    """
    tag_words: adds the occurrence to a word for each word in a sublist

    parameters:
        sents: list of lists, where a sublist represents a sentence

    returns: a list of lists, where elements of a sublist are tagged by their occurrence
    """
    for j in range(len(sents)):
        sent = sents[j]
        d = {}
        for i in range(len(sent)):
            word = sent[i]
            d[word] = d.get(word, 1)
            sent[i] = f"{word}_{d[word]}"
    return sents


def is_sublist(lst1, lst2):
    """
    is_sublist: true if the elements of lst1 are a subset of the elements of lst2
    """
    return set(lst1) <= set(lst2)


def is_super_pattern(ref_pat):
    """
    is_super_function: checks if pat is a super_pattern of ref_pat

    parameters:
        ref_pat: (list) an encoded pattern representing a sentence

    returns: True/False
    """
    target_len = len(ref_pat) + 1

    def _inner(pat, _):
        return (len(pat) == target_len
                and is_sublist(ref_pat, pat))

    return _inner


def divide_sentences(sentences, cur_pat, min_support):
    """
    divide_sentences: splits sentences into sentences that match the pattern, do not match pattern

    parameters:
        sentences: list of lists, where a sublist represents a sentence
        cur_pat: list of encoded words
        min_support (int)

    returns: list of tuples, each tuple has sentences matching pattern, not matching pattern
    """
    miner = PrefixSpan(sentences)
    options = miner.frequent(min_support, filter=is_super_pattern(cur_pat))
    if len(options) == 0:
        return []

    best_support, best_option = sorted(options, key=lambda e: e[0])[-1]
    has_pattern = [s for s in sentences if is_sublist(best_option, s)]
    residue = [s for s in sentences if s not in has_pattern]

    return [(has_pattern, best_option)] + divide_sentences(
        residue, cur_pat, min_support
    )


def find_leaf_patterns(sentences, cur_pat, *, min_support=1):
    """
    find_leaf_patterns: grow current pattern until reaching min_support

    parameters:
        sentences: list of lists, where a sublist represents a sentence
        cur_pat: (list) encoded words
        min_support (int)

    returns: tree structure describing how the pattern grew at each level
    """

    def pattern_added(entry):
        entry = entry.copy()
        added = list(
            set(Counter(entry["pattern"]).items()) - set(Counter(cur_pat).items())
        )
        entry["delta"] = added[0][0]
        return entry

    child_groups = divide_sentences(sentences, cur_pat, min_support)
    children = [
        find_leaf_patterns(s, p, min_support=min_support) for s, p in child_groups
    ]
    children = [pattern_added(c) for c in children]
    return {
        "delta": None,
        "pattern": cur_pat,
        "size": len(sentences),
        "children": children,
    }


def get_delta_index(child):
    """
    get_delta_index: finds and returns index of delta in child's pattern
    """
    child_p = child["pattern"]
    for j in range(len(child_p)):
        if child_p[j] == child["delta"]:
            return j


def rename_child(child, new_delta, index, patterns, encoder):
    """
    rename_child: update child delta, pattern and children information
    parameters:
        child: (dict)
        new_delta: (str) new delta value
        index: (int) where to insert new delta into child pattern
    returns: updated child
    """
    delta = [*patterns.encode([new_delta], encoder=encoder)][0]
    child["pattern"][index] = delta
    child["delta"] = delta
    child["children"] = []
    return child


def summarize_nodes(G, parent, nodes, num_exemplars):
    # Summarize children below min support and include a sample node
    summary = "+" + str(len(nodes) - num_exemplars) + "_" + nodes[0].split("_")[1]
    G.add_edge(parent, summary)
    G.nodes[summary]["summary_count"] = sum(G.nodes[remove]["count"] for remove in nodes[num_exemplars:])
    G.nodes[summary]["count"] = 1
    G.nodes[summary]["summary"] = True

    # create edges between summary node and the children's children
    for child in nodes[num_exemplars:]:
        for target in G.successors(child):
            if not G.has_edge(summary, target):
                G.add_edge(summary, target)
        G.remove_node(child)

    for node in nodes[:num_exemplars]:
        G.nodes[node]["exemplar"] = True

    return nodes[:num_exemplars] + [summary]


def prune_graph(G, min_support=3, num_exemplars=3):
    nodes = [*G.nodes()]
    for node in nodes:
        if G.has_node(node) and G.nodes[node]["count"] >= min_support:
            children_below = [child for child in G.successors(node)
                              if G.nodes[child]["count"] < min_support]

            if len(children_below) > num_exemplars:
                summarize_nodes(G, node, children_below, num_exemplars)
            else:
                for child in children_below:
                    G.nodes[child]["exemplar"] = True

    def _do_keep(data):
        return (data["count"] >= min_support
                or data.get("exemplar", False)
                or data.get("summary", False))

    keep = [node for node, data in G.nodes(data=True)
            if _do_keep(data)]

    return G.subgraph(keep)


def make_graph_from_leaves(tree, G, encoder, patterns, min_support=1):
    """
    make_graph_from_leaves: Finds leaf pattern and adds it to the tree
    parameters:
        tree: (list of dicts) a tree structure describing how sequential patterns grew
        G: (graph) of sequential leaf patterns
        encoder: used to decode words
        patterns: used to encode words
        min_support: (int)
    returns: (graph) of all leaf patterns
    """
    for child in tree["children"]:
        make_graph_from_leaves(child, G, encoder, patterns, min_support=min_support)

    if not tree["children"]:
        # Decode the pattern
        decoded = [encoder.decode(value=word) for word in tree["pattern"]]

        # Add nodes
        for word in decoded:
            if not G.has_node(word):
                G.add_node(word, count=tree["size"])

        # Add edges with weights
        for source, target in zip(decoded, decoded[1:]):
            # G.add_edge(source, target)
            if G.has_edge(source, target):
                G.edges[source, target]["weight"] += 1
            else:
                G.add_edge(source, target, weight=1)
    return G


def simplify_graph(G):
    """
    simplify_graph: Merges nodes with the same support
    parameters:
        G: (graph) of sequential leaf patterns
    returns: (graph) of all leaf patterns with merged nodes
    """
    for node in G.nodes():
        children = [*G.out_edges(nbunch=[node])]
        if (
            len(children) == 1
            and G.nodes[node]["count"] == G.nodes[children[0][1]]["count"]
        ):
            child = children[0][1]
            combined_name = node + " " + child

            mapping = {child: combined_name}
            G = nx.relabel_nodes(G, mapping)
            for parent in G.predecessors(node):
                initial_weight = G.get_edge_data(parent, node)['weight']
                G.add_edge(parent, combined_name, weight = initial_weight)
            G.remove_node(node)

            G = simplify_graph(G)
            return G
    return G


def build_sententree(sentences, *, min_support=2, num_exemplars=3, tag_with=tag_words_with_occurrence):
    split_sents = [sent.split() for sent in sentences]
    tagged_split_sents = tag_with(split_sents)
    encoder = patterns.Encoder()
    encoded_sents = [
        patterns.encode(sent, encoder=encoder) for sent in tagged_split_sents
    ]

    pattern_tree = find_leaf_patterns(encoded_sents, [], min_support=1)

    G = nx.DiGraph()
    make_graph_from_leaves(pattern_tree, G, encoder, patterns, min_support)
    pruned_G = prune_graph(G, min_support=min_support, num_exemplars=num_exemplars)
    G = simplify_graph(pruned_G)
    G.graph["num_encoded_sentences"] = len(encoded_sents)
    return G
