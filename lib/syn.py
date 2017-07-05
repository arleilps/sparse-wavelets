import random
import math

import networkx as nx
import numpy as np
from scipy import linalg


def synthetic_graph(size, num_edges, cut_sparsity, energy, balance, noise):
    """
        Return a graph "G" with the specified parameters
        Input:
            * size: number of vertices of "G"
            * num_edges: number of edges "G"
            * cut_sparsity: percentage of "remaining_edges" added the cut.
                If 1 all "remaining_edges" to add are chosen at random.
                If 0 all "remaining_edges" to add are chosen across the cut.
            * energy:
            * balance: if 0 size_part_a=0, if 2 size_part_b=0, for values
                near 1 the two partitions are balanced
            * noise: used to define "noise_std_dev"
        Output:
            * G: networkx graph
            * F: graph signal (np.array)
            * cut_num_edges: number of edges of the cut
    """

    # Define the partitions' sizes according to balance
    size_part_a = math.ceil(float(size * balance) / 2)
    size_part_b = size - size_part_a

    avg_a = np.sqrt((energy * size) / (size_part_a * size_part_b)) / 2
    avg_b = - avg_a

    F = []
    noise_std_dev = noise * avg_a

    for v in range(size):
        if v < size_part_a:
            F.append(random.gauss(avg_a, noise_std_dev))
        else:
            F.append(random.gauss(avg_b, noise_std_dev))
    F = np.array(F)

    G = nx.Graph()
    edges = set()
    # Build a line graph adding size-1 edges
    for v in range(size - 1):
        G.add_edge(v, v + 1)
        edges.add((v, v + 1))
    # The remaining edges are added according to cut_sparsity. The assignment
    # is done though repeated guesses. For 'num_edges' approaching
    # size*(size-1) it can take too long.
    remaining_edges = num_edges - len(G.edges())
    edges_within = int(cut_sparsity * remaining_edges)
    edges_across = remaining_edges - edges_within
    # Assign edges to the cut
    for e in range(edges_across):
        v1 = random.randint(0, size_part_a - 1)
        v2 = random.randint(size_part_a, size - 1)

        while (v1, v2) in edges or v1 == v2:
            v1 = random.randint(0, size_part_a - 1)
            v2 = random.randint(size_part_a, size_part_a + size_part_b - 1)

        G.add_edge(v1, v2)
        edges.add((v1, v2))

    for e in range(edges_within):

        while ((v1, v2) in edges or
               v1 == v2 or
               (v1 < size_part_a and v2 >= size_part_a)):
            v1 = random.randint(0, size - 1)
            v2 = random.randint(0, size - 1)
            # Edges are undirected
            if v1 > v2:
                tmp = v1
                v1 = v2
                v2 = tmp

        G.add_edge(v1, v2)
        edges.add((v1, v2))
        cut_num_edges = edges_across + 1

    return G, F, cut_num_edges


def compute_distances(center, graph):
    """
    Return a dict of distances
    key   : "vertex_id"
    value : shortest path length from "center" to "vertex_id"
    """
    distances = nx.shortest_path_length(graph, source=center)

    return distances


def compute_embedding(distances, radius, graph):
    B = np.array([distances[v] <= radius for v in graph.nodes()])

    return B


def generate_dyn_cascade(G, diam, duration, n):
    Fs = []

    for j in range(n):
        v = random.randint(0, len(G.nodes()) - 1)
        distances = compute_distances(G.nodes()[v], G)

        num_snaps = max(diam, duration)

        for i in range(num_snaps):
            # The radius r increases at each step
            r = int(i * math.ceil(float(diam) / duration))

            F = compute_embedding(distances, r, G)
            Fs.append(F)

    return np.array(Fs)


def generate_dyn_heat(G, s, jump, n):
    Fs = []
    L = nx.normalized_laplacian_matrix(G)
    L = L.todense()
    F0s = []
    seeds = []

    for i in range(s):
        F0 = np.zeros(len(G.nodes()))
        v = random.randint(0, len(G.nodes()) - 1)
        seeds.append(v)
        F0[v] = len(G.nodes())
        F0s.append(F0)

    Fs.append(np.sum(F0s, axis=0))

    for j in range(n):
        FIs = []
        for i in range(s):
            FI = np.multiply(linalg.expm(-j * jump * L), F0s[i])[:, seeds[i]]
            FIs.append(FI)

        Fs.append(np.sum(FIs, axis=0))

    return np.array(Fs)[1:]


def generate_dyn_gaussian_noise(G, n):
    Fs = np.random.rand(n, len(G.nodes()))

    return Fs


def generate_dyn_bursty_noise(G, n):
    Fs = []
    bursty_beta = 1
    non_bursty_beta = 1000
    bursty_bursty = 0.7
    non_bursty_non_bursty = 0.9
    bursty = False

    for j in range(n):
        r = random.random()

        if not bursty:
            if r > non_bursty_non_bursty:
                bursty = True
        else:
            if r > bursty_bursty:
                bursty = False

        if bursty:
            F = np.random.exponential(bursty_beta, len(G.nodes()))
        else:
            F = np.random.exponential(non_bursty_beta, len(G.nodes()))

        Fs.append(F)

    return np.array(Fs)


def generate_dyn_indep_cascade(G, s, p):
    Fs = []

    seeds = np.random.choice(len(G.nodes()), s, replace=False)

    F0 = np.zeros(len(G.nodes()))

    ind = {}
    i = 0

    for v in G.nodes():
        ind[v] = i
        i = i + 1

    for s in seeds:
        F0[s] = 2.0

    while True:
        F1 = np.zeros(len(G.nodes()))
        new_inf = 0
        for v in G.nodes():
            if F0[ind[v]] > 1.0:
                for u in G.neighbors(v):
                    r = random.random()
                    if r <= p and F0[ind[u]] < 1.0:
                        F1[ind[u]] = 2.0
                        new_inf = new_inf + 1
                F1[ind[v]] = 1.0
                F0[ind[v]] = 1.0
            elif F0[ind[v]] > 0.0:
                F1[ind[v]] = 1.0

        Fs.append(F0)

        if new_inf == 0 and len(Fs) > 1:
            break

        F0 = np.copy(F1)

    return np.array(Fs)


def generate_dyn_linear_threshold(G, s):
    Fs = []

    seeds = np.random.choice(len(G.nodes()), s, replace=False)

    F0 = np.zeros(len(G.nodes()))
    thresholds = np.random.uniform(0.0, 1.0, len(G.nodes()))

    ind = {}
    i = 0

    for v in G.nodes():
        ind[v] = i
        i = i + 1

    for s in seeds:
        F0[s] = 1.0

    while True:
        F1 = np.zeros(len(G.nodes()))
        new_inf = 0
        for v in G.nodes():
            if F0[ind[v]] < 1.0:
                n = 0
                for u in G.neighbors(v):
                    if F0[ind[u]] > 0:
                        n = n + 1

                if (float(n) / len(G.neighbors(v))) >= thresholds[ind[v]]:
                    F1[ind[v]] = 1.0
                    new_inf = new_inf + 1
            else:
                F1[ind[v]] = 1.0

        Fs.append(F0)

        if new_inf == 0 and len(Fs) > 1:
            break

        F0 = np.copy(F1)

    return np.array(Fs)
