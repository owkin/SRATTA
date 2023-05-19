"""
The goal of this function is to be able to gather samples by center. We see
the sample as nodes of a graph and an edge is drawn as soon as two samples are in
the same center. When then look for the connected component of this graph
"""

import networkx
from networkx.algorithms.components.connected import connected_components


def to_graph(graph):
    graph_object = networkx.Graph()
    for part in graph:
        # each sublist is a bunch of nodes
        graph_object.add_nodes_from(part)
        # it also imlies a number of edges:
        graph_object.add_edges_from(to_edges(part))
    return graph_object


def to_edges(graph):
    """
    treat `graph` as a Graph and returns it's edges
    to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(graph)
    last = next(it)

    for current in it:
        yield last, current
        last = current


def connect_components_from_list(graph):
    graph = to_graph(graph)
    return connected_components(graph)
