"""
This module read graph's info from files with the following format:
input_graph_name has info about edges.
Row format: "node_A, node_B[, edge_weight]"
input_data_name has info about the graph signal.
Row format : "node_id, node_value"
"""

import networkx as nx
import numpy as np


def read_graph(input_graph_name, input_data_name):
    """
        Read graph from file.
        Input:
            * input_graph_name has info about edges.
                Row format: "node_A, node_B[, edge_weight]"
            * input_data_name has info about the graph signal.
                Row format : "node_id, node_value"
        Output:
            * networkx graph
    """

    # Reading input data
    input_data = open(input_data_name, 'r')
    graph_signal = {}

    for line in input_data:
        line = line.rstrip()
        node, value = line.rsplit(',')
        value = float(value)
        graph_signal[node] = value

    input_data.close()

    # Reading graph data
    G = nx.Graph()
    input_graph = open(input_graph_name, 'r')

    for line in input_graph:
        line = line.rstrip()
        node_A, node_B = line.rsplit(',')[:2]
        # Note that the edge weight is always set to 1
        # even when provided available in input_graph_name
        if node_A in graph_signal and node_B in graph_signal:
            G.add_edge(node_A, node_B, weight=1.)

    input_graph.close()

    # Extracting largest connected component from graph
    Gcc = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)

    G = Gcc[0]

    # Setting the graph_signal as node attribute
    for node, value in graph_signal.items():
        if node in G:
            G.node[node]["value"] = value

    return G


def read_values(input_data_name, G):
    """
        Read the graph signal from file
        Input:
            * input_data_name has info about the graph signal.
                Row format : "node_id, node_value"
            * G: networkx graph
        Output:
            * F: normalized node values array, ordered by G.nodes()
    """
    graph_signal = {}
    input_data = open(input_data_name, 'r')

    # Reading file
    for line in input_data:
        line = line.rstrip()
        node, value = line.rsplit(',')
        graph_signal[node] = float(value)

    input_data.close()

    F = []
    for node in G.nodes():
        if node in graph_signal:
            F.append(graph_signal[node])
        else:
            F.append(0.)

    # Normalization
    F = np.array(F)
    F = F / np.max(F)
    F = F - np.mean(F)

    return F
