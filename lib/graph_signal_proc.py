import math
from collections import deque

import networkx as nx
import numpy as np
from numpy import dot, diag
import scipy.optimize
from scipy import linalg
import scipy.fftpack


def compute_eigenvectors_and_eigenvalues(L):
    """
        Computes eigenvectors and eigenvalues of the matrix L
        Input:
            * L: matrix
        Output:
            * U: eigenvector matrix, one vector/column, sorted by corresponding
                 eigenvalue
            * lamb: eigenvalues, sorted in increasing order
    """
    lamb, U = linalg.eig(L)

    idx = lamb.argsort()
    lamb = lamb[idx]
    U = U[:, idx]

    return U, lamb


def g(x):
    """
        Wavelet generating kernel, see Hammond, D. K.,Vandergheynst, P.,
        & Gribonval, R. (2011). "Wavelets on graphs via spectral graph theory".
        Input:
            * x
        Output:
            * kernel of x
    """
    a = 2
    b = 2
    x_1 = 1
    x_2 = 2

    if x < x_1:
        return pow(x_1, -a) * pow(x, a)
    elif x <= x_2 and x >= x_1:
        return -5 + 11 * x - 6 * pow(x, 2) + pow(x, 3)
    else:
        return pow(x_2, b) * pow(x, -b)


def comp_gamma():
    """
        In Hammond, D. K.,Vandergheynst, P.,& Gribonval, R. (2011).
        "Wavelets on graphs via spectral graph theory" gamma is a parameter
        used to determine the scaling function h. It is such that h(0) = max(g)
        Input:
                * None
        Output:
                * Gamma function (array)
    """
    # fminbound finds the minimum within the optimization bounds
    xopt = scipy.optimize.fminbound(lambda x: -g(x), 1, 2)
    return xopt


def h(x, gamma, lamb_max, K):
    """
        Scaling function see Hammond, D. K.,Vandergheynst, P.,
        & Gribonval, R. (2011). "Wavelets on graphs via spectral graph theory".
        Input:
                * x
                * gamma
                * lamb_max: upper bound spectrum
                * K: normalization
        Output:
                * value of scaling function
    """
    lamb_min = float(lamb_max) / K
    return gamma * math.exp(-pow(float(x / (lamb_min * 0.6)), 4))


def comp_scales(lamb_max, K, J):
    r"""
        Computes wavelet scales see Hammond, D. K.,Vandergheynst, P.,
        & Gribonval, R. (2011). "Wavelets on graphs via spectral graph theory".
        Input:
            * lamb_max: upper bound spectrum
            * K: desired ratio for lambda_max / lambda_min
            * J: number of scales
        Output:
            * scales array
    """
    lamb_min = float(lamb_max) / K
    s_min = float(1) / lamb_max
    s_max = float(2) / lamb_min

    return np.exp(np.linspace(math.log(s_max), math.log(s_min), J))


def graph_low_pass(lamb, U, T, gamma, lamb_max, K):
    """
        Low-pass spectral filter (square matrix).
        See "The emerging field of signal processing on graph"
        Input:
            * lamb: eigenvalues
            * U: eigenvector matrix
            * N: number of nodes
            * T: wavelet scales
            * gamma: scaling function parameter
            * lamb_max: upper-bound spectrum
            * K: normalization
        Output:
            * s: Low-pass filter as a N x N matrix
    """

    h_vector = [h(T[-1] * l, gamma, lamb_max, K) for l in lamb]

    return dot(U, dot(diag(h_vector), U.T))


def graph_wavelets(lamb, U, N, T):
    """
        Graph wavelets.
        Input:
            * lamb: eigenvalues
            * U: eigenvector matrix
            * N: number of nodes
            * T: wavelet scales
        Output:
            * w: wavelets as a len(T) x N x N matrix
    """

    w = []

    for t in range(len(T)):
        g_vector = [g(T[t] * l) for l in lamb]
        w.append(dot(U, dot(diag(g_vector), U.T)))

    return np.asarray(w)


def graph_fourier(F, U):
    """
        Graph Fourier transform.
        Input:
            * F: Signal in the vertex domain
            * U: Eigenvectors matrix
        Ouput:
            * F_hat: Signal in the graph spectral domain
    """
    F_hat = []

    for i in range(0, len(U)):
        F_hat.append(dot(F, U[:, i]))

    F_hat = np.array(F_hat)

    return F_hat


def graph_fourier_inverse(F_hat, U):
    """
        Graph Fourier inverse:
        Input:
            * F_hat: Signal in the graph spectral domain
            * U: Eigenvectors matrix
        Output:
            * F: Signal in the vertex domain
    """
    F = np.zeros(U.shape[0])
    for v in range(U.shape[0]):
        for u in range(U.shape[1]):
            F[v] = F[v] + (F_hat[u] * U[v][u]).real

    return F


def hammond_wavelet_transform(w, s, T, F):
    r"""
        Hammond wavelet transform.
        Input:
            * w: wavelets
            * s: low-pass wavelet (scaling function)
            * T: wavelet scales
            * F: graph signal
        Output:
            * C: Hammond's wavelet transform. (len(T) + 1) x len(F)
                 matrix of transform coefficients
    """
    C = []

    for i in range(len(T)):
        # Each wavelet is represented by an N x N matrix
        C.append(dot(F, w[i].T))
    # Append output of scaling function application at the end
    C.append(dot(F, s.T))

    return np.asarray(C)


def hammond_wavelets_inverse(w, s, C):
    r"""
        Hammond's wavelet inverse.
        Input:
            * w: wavelets
            * s: low-pass wavelet (scaling function)
            * C: Hammond's wavelet transform. (len(T) + 1) x len(F)
                 matrix of transform coefficients
        Output:
            * F: Reconstructed signal in the vertex domain
    """
    nC = np.ravel(C)
    Wc = np.append(w, np.array([s]), axis=0)
    nWc = Wc.reshape(Wc.shape[0] * Wc.shape[1], Wc.shape[2])
    # Search a least square solution F, solving:
    # nWc F = nC
    F = np.linalg.lstsq(nWc, nC)[0]

    return F


class Node(object):
    """
        Generic tree-structure used for hierarchical transforms.
    """

    def __init__(self, data):
        """
            Input:
                * data: Anything to be stored in a node.
                    Usually only leaf nodes have data != None
                    data != None often used as stopping condition
        """
        self.data = data
        self.children = []
        self.diffs = []
        # Level on the tree. The root has scale = 0
        self.scale = 0
        # count: number of leaves (data != None) of its subtree
        if data is None:
            self.count = 0
        else:
            self.count = 1

    def __str__(self):
        descr = "Node id: {}, data: {}, scale: {}, count: {}"
        return descr.format(id(self), self.data, self.scale, self.count)

    def __repr__(self):
        return self.__str__()

    def add_child(self, obj):
        """
            Adds obj as a child to a node.
            Input:
                * obj: anything
        """
        obj.scale = self.scale + 1
        self.children.append(obj)
        self.count = self.count + obj.count


def set_counts(tree):
    """
        Input:
            * tree: tree node
        Output:
            * count: count for the tree node
    """
    if tree.data is not None:
        tree.count = 1
        return 1
    else:
        count = 0
        for c in tree.children:
            count = count + set_counts(c)

        tree.count = count

        return count


def set_fiedler_method(method):
    # Set method for Fiedler vector computation
    global _method
    _method = method


def sweep(x, G):
    """
        Sweep algorithm for ratio-cut (2nd eigenvector of the Laplacian).
        Based on vector x.
        Input:
            * x: vector
            * G: graph
        Output:
            * vec: indicator vector
    """
    sorted_x = np.argsort(x)
    part_one = set()
    N = nx.number_of_nodes(G)
    best_val = N - 1
    edges_cut = 0
    nodes_list = list(G.nodes())

    for i in range(N - 1):
        part_one.add(nodes_list[sorted_x[i]])

        for v in G.neighbors(nodes_list[sorted_x[i]]):
            if v not in part_one:
                edges_cut = edges_cut + 1
            else:
                edges_cut = edges_cut - 1

        den = len(part_one) * (N - len(part_one))

        if den > 0:
            val = float(edges_cut) / den
            if val <= best_val:
                best_cand = i
                best_val = val

    vec = np.ones(nx.number_of_nodes(G))

    for i in range(x.shape[0]):
        if i <= best_cand:
            vec[sorted_x[i]] = -1.

    return vec


def separate_lcc(G, G0):
    """
        Separate vertices in G0 (LCC) from the rest in G returning
        an indicator vector.
        Input:
            * G: Graph
            * G0: Subgraph
        Output:
            * x: indicator vector
    """

    return np.array([-1. if v in G0 else 1. for v in G.nodes()])


def ratio_cut(G):
    """
        Computes ratio-cut of G based on second eigenvector of the Laplacian.
        Input:
            * G: Graph
        Output:
            * x: Indicator vector
    """

    Gcc = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)
    G0 = Gcc[0]

    if nx.number_of_nodes(G) == nx.number_of_nodes(G0):
        scipy.random.seed(1)
        x = nx.fiedler_vector(G, method=_method, tol=1e-5)
        x = sweep(x, G)
    else:
        # In case G is not connected
        x = separate_lcc(G, G0)
    return np.array(x)


def get_subgraphs(G, cut):
    """
        Return the two subgraphs as two lists of nodes
        Input:
            * G: Original graph
            * cut: cut indicator vector
        Output:
            * G1: subgraph 1
            * G2: subgraph 2
    """
    G1 = nx.Graph()
    G2 = nx.Graph()
    i = 0
    P1 = []
    P2 = []
    for v in G.nodes():
        if cut[i] < 0:
            P1.append(v)
        else:
            P2.append(v)
        i = i + 1

    G1 = G.subgraph(P1)
    G2 = G.subgraph(P2)

    return G1, G2


def rc_recursive(node, G, ind):
    """
        Recursively computes ratio-cut.
        The leaves store, as data, the integer returned by ind for the
        inserted node.
        Input:
            * node: tree node
            * G: graph
            * ind: index with unique integers as values
                (see ratio_cut_hierarchy for definition)
        Output:
            * none
    """
    if nx.number_of_nodes(G) < 3:
        n = Node(None)
        n.add_child(Node(ind[list(G.nodes())[0]]))
        n.add_child(Node(ind[list(G.nodes())[1]]))
        node.add_child(n)
    else:
        C = ratio_cut(G)

        (G1, G2) = get_subgraphs(G, C)

        if nx.number_of_nodes(G1) > 1:
            l = Node(None)
            rc_recursive(l, G1, ind)
            node.add_child(l)
        else:
            l = Node(ind[list(G1.nodes())[0]])
            node.add_child(l)

        if nx.number_of_nodes(G2) > 1:
            r = Node(None)
            rc_recursive(r, G2, ind)
            node.add_child(r)
        else:
            r = Node(ind[list(G2.nodes())[0]])
            node.add_child(r)


def ratio_cut_hierarchy(G, method='lobpcg'):
    """
        Computes ratio-cut hierarchy for a graph.
        The leaves store, as data, the integer returned by ind for the
        inserted node.
        Input:
            * G: graph
            * method: method for Fiedler vector computation.
                The default value is 'lobpcg' however 'tracemin_lu' seems
                faster and it appears to give more stable results when used
                with PYTHONHASHSEED set to a constant value.
        Output:
            * root: tree root
            * ind: index with unique integers as values

    """
    global _method
    _method = method

    ind = {v: i for i, v in enumerate(G.nodes())}

    root = Node(None)

    rc_recursive(root, G, ind)

    return root, ind


def compute_coefficients(tree, F):
    """
        Compute tree coefficients for Gavish's transform.
        Input:
            * tree: tree
            * F: graph signal
        Output:
            * None
    """
    if tree.data is None:
        tot = 0
        count = 0
        for i, child in enumerate(tree.children):
            compute_coefficients(child, F)
            tot += child.avg * child.count
            count += child.count

            if i > 0:
                tree.diffs.append(2 * child.count *
                                  (child.avg - float(tot) / count))
        tree.avg = float(tot) / tree.count
    else:
        tree.avg = F[tree.data]


def reconstruct_values(tree, F):
    """
            Reconstruct values for Gavish's transform based on a tree.
            Input:
                    * tree: tree
                    * F: graph signal
            Output:
                    * None
    """
    if tree.data is None:
        tot = tree.avg * tree.count
        count = tree.count
        for i in reversed(range(len(tree.children))):
            if i == 0:
                tree.children[i].avg = tot / tree.children[i].count
                reconstruct_values(tree.children[i], F)
            else:
                tree.children[i].avg = float(tot) / count + 0.5 * \
                    float(tree.diffs[i - 1]) / tree.children[i].count
                reconstruct_values(tree.children[i], F)
                count = count - tree.children[i].count
                tot = tot - tree.children[i].avg * tree.children[i].count

    else:
        F[tree.data] = tree.avg


def clear_tree(tree):
    """
        Clear tree info.
        tree.count is kept
        Input:
            * tree
        Output:
            * None
    """
    tree.avg = 0
    tree.diffs = []

    if tree.data is None:
        for i in range(len(tree.children)):
            clear_tree(tree.children[i])


def get_coefficients(tree, wtr):
    """
        Recover wavelet coefficients from the wavelet tree.
        Input:
            * tree
            * wtr: list of wavelet coefficients
        Output:
            * None
    """
    Q = deque()
    wtr.append(tree.count * tree.avg)

    Q.append(tree)

    while len(Q) > 0:
        node = Q.popleft()

        for j in range(len(node.diffs)):
            wtr.append(node.diffs[j])

        for i in range(len(node.children)):
            Q.append(node.children[i])


def set_coefficients(tree, wtr):
    """
        Sets wavelet tree coefficients.
        Input:
            * tree
            * wtr: list of wavelet coefficients
    """
    Q = deque()
    tree.avg = float(wtr[0]) / tree.count
    p = 1
    Q.append(tree)

    while len(Q) > 0:
        node = Q.popleft()

        for j in range(len(node.children) - 1):
            node.diffs.append(wtr[p])
            p += 1

        for i in range(len(node.children)):
            Q.append(node.children[i])


def gavish_wavelet_transform(tree, G, F):
    """
        Gavish's wavelet transform.
        Input:
            * tree
            * ind: vertex index v : unique integer
            * G: graph
            * F: graph signal
        Output:
            * wtr: wavelet transform.
    """
    wtr = []
    clear_tree(tree)
    compute_coefficients(tree, F)
    get_coefficients(tree, wtr)
    return np.array(wtr)


def gavish_wavelet_inverse(tree, ind, G, wtr):
    """
        Gavish's wavelet inverse.
        Input:
            * tree
            * ind: vertex index v: unique integer
            * G: graph
            * wtr: wavelet transform
        Output:
            * F: wavelet inverse
    """
    F = []

    for i in range(len(G.nodes())):
        F.append(0)

    clear_tree(tree)
    set_coefficients(tree, wtr)
    reconstruct_values(tree, F)

    return np.array(F)
