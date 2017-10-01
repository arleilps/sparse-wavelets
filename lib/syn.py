import math
import random

import networkx as nx
import numpy as np


def synthetic_graph(size, num_edges, sparsity, energy, balance, noise,
                    seed=None):
    r"""
        Build a synthetic connected graph and a graph signal starting from two
        sets of vertices. These sets have sizes depending on size and balance.
        Input:
            * size: number of vertices
            * num_edges: number of edges
            * sparsity: higher values penalize the creation of edges
                having vertices in different sets
            * energy: is the coefficient's energy obtained from the generated
                graph signal and using as partitions the two vertex sets
            * balance: takes values in (0,2). If 1 the two sets from which the
                synthetic graph is built have similar size.
            * noise: part of the std dev used for generating the noisy signal
            * seed: optional seed for the random module
        Output:
            * G: connected graph
            * np.array(F): graph signal
            * edges_accross + 1: number of edges having vertices in different
                sets
    """
    if seed:
        random.seed(seed)

    if balance <= 0 and balance >= 2:
        raise ValueError("'balance' should be in (0,2)")

    if num_edges < size:
        raise ValueError("The graph returned by synthetic_graph() is assumed "
                         + "to be connected: choose num_edges >= size - 1")

    size_part_a = int(math.ceil(size * balance / 2.))
    size_part_b = size - size_part_a

    if size_part_a == 0:
        raise ValueError("'size_part_a'==0 try bigger values " +
                         "for 'size' or 'balance'")
    if size_part_b == 0:
        raise ValueError("'size_part_b'==0 try a bigger value " +
                         "for 'size' or a smaller value for 'balance'")

    avg_a = np.sqrt(float(energy * size) /
                    (size_part_a * size_part_b)) / 2.
    avg_b = - avg_a

    F = []
    for v in range(size):
        if v < size_part_a:
            F.append(random.gauss(avg_a, noise * avg_a))
        else:
            F.append(random.gauss(avg_b, noise * avg_a))

    G = nx.Graph()

    edges_set = set({})
    for v in range(size - 1):
        G.add_edge(v, v + 1)
        edges_set.add((v, v + 1))

    remaining_edges = num_edges - len(G.edges())
    edges_accross = int((size_part_a * size_part_b * (1. - sparsity)
                         * remaining_edges) / (size * (size - 1)))
    edges_within = remaining_edges - edges_accross

    for e in range(edges_accross):
        v1 = random.randint(0, size_part_a - 1)
        v2 = random.randint(size_part_a, size - 1)

        while (v1, v2) in edges_set or v1 == v2:
            v1 = random.randint(0, size_part_a - 1)
            v2 = random.randint(size_part_a, size_part_a + size_part_b - 1)

        G.add_edge(v1, v2)
        edges_set.add((v1, v2))

    for e in range(edges_within):
        v1 = random.randint(0, size - 1)
        v2 = random.randint(0, size - 1)

        if v1 > v2:
            tmp = v1
            v1 = v2
            v2 = tmp

        while ((v1, v2) in edges_set or v1 == v2 or
                (v1 < size_part_a and v2 >= size_part_a) or
                (v1 >= size_part_a and v2 < size_part_a)):
            v1 = random.randint(0, size - 1)
            v2 = random.randint(0, size - 1)

            if v1 > v2:
                tmp = v1
                v1 = v2
                v2 = tmp

        G.add_edge(v1, v2)
        edges_set.add((v1, v2))

    return G, np.array(F), edges_accross + 1
