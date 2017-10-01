import math

import networkx as nx
import numpy as np
from numpy import dot, diag, sqrt
from numpy.linalg import eigh
import scipy

from lib.graph_signal_proc import Node
import lib.graph_signal_proc as gsp


def sweep_opt(x, F, G, k, ind):
    """
        Sweep algorithm for sparse wavelets.
        Input:
            * x: continuous indicator vector
            * F: graph signal
            * G: graph
            * k: max number of edges to be cut
            * ind: vertex index v: unique integer
        Output:
            * vec: indicator vector
            * best_energy: max energy value
            * best_cut_size: number of edges in the returned cut
    """
    sorted_x = np.argsort(x)
    part_one = set()
    N = nx.number_of_nodes(G)
    best_energy = 0.
    cut_size = 0
    best_cut_size = 0
    sum_one = 0     # sum of the graph signal values for the nodes in part_one
    sum_two = 0     # sum of the graph signal values for the nodes in part_two

    for v in G.nodes():
        sum_two += F[ind[v]]

    nodes_list = list(G.nodes())

    for i in range(N):
        part_one.add(nodes_list[sorted_x[i]])
        sum_one += F[ind[nodes_list[sorted_x[i]]]]
        sum_two -= F[ind[nodes_list[sorted_x[i]]]]

        for v in G.neighbors(nodes_list[sorted_x[i]]):
            if v not in part_one:
                cut_size += 1
            else:
                cut_size -= 1

        den = N * len(part_one) * (N - len(part_one))
        if den > 0:
            energy = math.pow(sum_one * (N - len(part_one)) - sum_two *
                              len(part_one), 2) / den
        else:
            energy = 0

        if energy >= best_energy and cut_size <= k:
            best_cand = i
            best_energy = energy
            best_cut_size = cut_size

    vec = np.ones(nx.number_of_nodes(G))

    for i in range(x.shape[0]):
        if i <= best_cand:
            vec[sorted_x[i]] = -1.

    return vec, best_energy, best_cut_size


###############################################################################
#  List of functions used only by the SWT:                                    #
#  - sqrtmi()                                                                 #
#  - spectral_cut()                                                           #
#  - complete_graph_laplacian                                                 #
#  - weighted_adjacency_complete                                              #
#  - one_d_search                                                             #
###############################################################################


def sqrtmi(mat):
    """
        Computes the square-root inverse of a matrix.
        Input:
            * mat: matrix
        Output:
            * square root inverse
    """
    eigvals, eigvecs = eigh(mat)
    eigvecs = eigvecs[:, eigvals > 0]
    eigvals = eigvals[eigvals > 0]

    return dot(eigvecs, dot(diag(1. / sqrt(eigvals)), eigvecs.T))


def spectral_cut(CAC, L, C, A, start, F, G, beta, k, ind):
    """
        Spectral cut implementation.
        Input:
            * CAC: C*A*C where C is the Laplacian of a complete graph and
                 A is a pairwise squared difference matrix
            * L: graph laplacian matrix
            * C: laplacian complete graph
            * A: pairwise squared difference matrix
            * start: initialization
            * beta: regularization parameter
            * k: max edges cut
            * ind: vertex index vertex: unique integer
        Output:
            * res: dictionary with following fields:
                - x: indicator vector
                - size: number of edges cut
                - energy: cut energy
    """
    isqrtCL = sqrtmi(C + beta * L)
    M = np.dot(np.dot(isqrtCL, CAC), isqrtCL)

    (eigvals, eigvecs) = scipy.linalg.eigh(M, eigvals=(0, 0))
    x = np.asarray(np.dot(eigvecs[:, 0], isqrtCL))[0, :]

    res = {}
    res["x"], res["energy"], res["size"] = sweep_opt(x, F, G, k, ind)

    return res


def complete_graph_laplacian(n):
    """
        Laplacian of a complete graph with n vertices.
        Input:
            * n: size
        Output:
            * C: Laplacian
    """
    C = np.ones((n, n))
    C = -1 * C
    D = np.diag(np.ones(n))
    C = (n) * D + C

    return C


def weighted_adjacency_complete(G, F, ind):
    """
        Computes weighted adjacency complete matrix (w(v)-w(u))^2
        Input:
            * G: graph
            * F: graph signal
            * ind: vertex index vertex: unique integer
        Output:
            * A: nxn matrix
    """
    A = []
    for v in G.nodes():
        A.append([])
        for u in G.nodes():
            A[-1].append(pow(F[ind[v]] - F[ind[u]], 2))

    return np.array(A)


def one_d_search(G, F, k, ind):
    """
        Cut computation. Perform 1-D search for beta using golden search.
        Input:
            * G: graph
            * F: graph signal
            * k: max edges to be cut
            * n: number of chebyshev polynomials
            * ind: vertex index vertex: unique integer
        Output:
            * cut
    """
    C = complete_graph_laplacian(nx.number_of_nodes(G))
    A = weighted_adjacency_complete(G, F, ind)
    CAC = np.dot(np.dot(C, A), C)
    start = np.ones(nx.number_of_nodes(G))
    L = nx.laplacian_matrix(G).todense()

    # Upper and lower bounds for beta
    gr = (math.sqrt(5) - 1) / 2
    a = 0.
    b = 1000.
    c = b - gr * (b - a)
    d = a + gr * (b - a)

    # Tolerance
    tol = 1.

    resab = {}
    resab["size"] = k + 1

    # golden search
    while abs(c - d) > tol or resab["size"] > k:
        resc = spectral_cut(CAC, L, C, A, start, F, G, c, k, ind)
        resd = spectral_cut(CAC, L, C, A, start, F, G, d, k, ind)

        if resc["size"] <= k and (resc["energy"] > resd["energy"]):
            start = resc["x"]
            b = d
            d = c
            c = b - gr * (b - a)
        elif resd["size"] <= k and (resc["energy"] < resd["energy"]):
            start = resd["x"]
            a = c
            c = d
            d = a + gr * (b - a)
        else:
            start = resc["x"]
            a = c
            c = d
            d = a + gr * (b - a)

        resab = spectral_cut(CAC, L, C, A, start, F, G, (b + a) / 2, k, ind)

    return resab


###############################################################################
#  List of functions used only by the FSWT:                                   #
#  - trans()                                                                  #
#  - isqrt()                                                                  #
#  - coef()                                                                   #
#  - chebyshev_approx_2d()                                                    #
#  - chebyshev_approx_1d()                                                    #
#  - fast_cac()                                                               #
#  - power_method()                                                           #
#  - cheb_spectral_cut()                                                      #
#  - fast_search()                                                            #
###############################################################################


def trans(L, min_v, max_v):
    """
        Chebyshev polynomial translation.
        Input:
            * L: Laplacian matrix
            * min_v: lower bound
            * max_v: upper bound
        Output:
            * translation as tuple of a sparse matrix and a scalar
    """
    return (2. / (max_v - min_v)) * L, -(float(max_v + min_v) /
                                         (max_v - min_v))


def isqrt(k, beta, min_v, max_v, x):
    """
        Inverse square root function to be integrated in
        Chebyshev polynomial computation.
        Input:
            * k: coefficient number
            * beta: regularization parameter
            * min_v: lower bound
            * max_v: upper bound
            * x: function variable
        Output:
            * function value
    """
    y = 0.5 * (math.cos(x) * float(max_v - min_v) + (max_v + min_v))

    return math.cos(k * x) * (1. / math.sqrt(beta * y))


def coef(k, beta, min_v, max_v):
    """
        Get the k-th Chebyshev polynomial coefficient for the inverse
        square root
        Input:
            * k: coefficient number
            * min_v: lower bound
            * max_v: upper bound
        Output:
            * coefficient
    """
    return (2. / math.pi) * (scipy.integrate.quad(
        lambda x: isqrt(k, beta, min_v, max_v, x), 0., math.pi)[0])


def chebyshev_approx_2d(n, beta, X, L):
    """
        Approximates sqrt((beta * L)^+) * X * sqrt((beta * L)^+) using
        Chebyshev polynomials (twice)
        Input:
            * n: number of polynomials
            * beta: regularization parameter
            * X: matrix
            * L: Laplacian matrix
        Output:
            * P2: approximation
    """
    N = L.shape[0]
    max_v = beta * N
    min_v = 1
    # ts1 is a sparse matrix, ts2 is a scalar
    ts1, ts2 = trans(L, min_v, max_v)
    # Recurrence relation for Chebyshev polynomials
    # Tk = 2 * y * Tk_1 - Tk_2
    # 0,5 * c_n0 * x + sum_{k=1}^{inf} (c_nk * Tk) * x
    P1 = 0.5 * coef(0, beta, min_v, max_v) * X
    tkm2 = X
    tkm1 = scipy.sparse.csr_matrix.dot(ts1, X) + ts2 * X
    P1 = P1 + coef(1, beta, min_v, max_v) * tkm1

    for i in range(2, n):
        Tk = 2. * (scipy.sparse.csr_matrix.dot(ts1, tkm1) + ts2 * tkm1) - tkm2
        P1 = P1 + coef(i, beta, min_v, max_v) * Tk
        tkm2 = tkm1
        tkm1 = Tk

    P1 = P1.transpose()
    P2 = 0.5 * coef(0, beta, min_v, max_v) * P1
    tkm2 = P1
    tkm1 = scipy.sparse.csr_matrix.dot(ts1, P1) + ts2 * P1

    P2 = P2 + coef(1, beta, min_v, max_v) * tkm1

    for i in range(2, n):
        Tk = 2. * (scipy.sparse.csr_matrix.dot(ts1, tkm1) + ts2 * tkm1) - tkm2
        P2 = P2 + coef(i, beta, min_v, max_v) * Tk
        tkm2 = tkm1
        tkm1 = Tk

    return P2


def chebyshev_approx_1d(n, beta, x, L):
    """
        Approximates x*sqrt((beta * L)^+) using Chebyshev polynomials.
        Input:
            * n: number of polynomials
            * beta: regularization parameter
            * x: vector
            * L: graph Laplacian
        Output:
            * P: approximation

    """
    N = L.shape[0]
    max_v = beta * N
    min_v = 1
    # ts1 is a sparse matrix, ts2 is a scalar
    ts1, ts2 = trans(L, min_v, max_v)
    # Recurrence relation for Chebyshev polynomials
    # Tk = 2 * y * Tk_1 - Tk_2
    # 0,5 * c_n0 * x + sum_{k=1}^{inf} (c_nk * Tk) * x
    Tk_2 = x
    Tk_1 = scipy.sparse.csr_matrix.dot(ts1, x) + ts2 * x
    P = 0.5 * coef(0, beta, min_v, max_v) * x
    P += coef(1, beta, min_v, max_v) * Tk_1
    """(2./ (max_v - min_v)) * L, -(float(max_v + min_v) /
                                         (max_v - min_v))"""
    for i in range(2, n):
        Tk = 2. * (scipy.sparse.csr_matrix.dot(ts1, Tk_1) + ts2 * Tk_1) - Tk_2
        P += coef(i, beta, min_v, max_v) * Tk
        Tk_2 = Tk_1
        Tk_1 = Tk

    return P


def fast_cac(G, F, ind):
    """
        Computes product C*A*C, where C is the Laplacian of a complete graph
        and A is a pairwise squared difference matrix.
        Input:
            * G: graph
            * F: graph signal
            * ind: vertex index v: unique integer
        Output:
            * CAC: matrix product
    """
    sorted_F = np.array([F[ind[v]] for v in G.nodes()])
    return -2 * math.pow(len(sorted_F), 2) * np.outer(sorted_F, sorted_F)


def power_method(mat, start, maxit):
    """
        Power method implementation.
        Input:
            * mat: matrix
            * start: initialization
            * maxit: number of iterations
        Output:
            * vec: largest eigenvector of mat
    """
    vec = np.copy(start)
    vec = vec / np.linalg.norm(vec)

    for i in range(maxit):
        vec = np.dot(vec, mat)

    return vec


def cheb_spectral_cut(CAC, start, F, G, beta, k, n, ind):
    """
        Fast spectral cut implementation using chebyshev polynomials.
        Input:
            * CAC: C*A*C where C is the Laplacian of a complete graph and
                A is a pairwise squared difference matrix
            * start: initialization
            * F: graph signal
            * G: graph
            * beta: regularization parameter
            * k: max edges cut
            * n: number of polynomials
            * ind: vertex index vertex: unique integer
        Output:
            * res: dictionary with following fields:
                - x: indicator vector
                - size: number of edges cut
                - energy: cut energy
    """
    L = nx.laplacian_matrix(G)
    M = chebyshev_approx_2d(n, beta, CAC, L)

    eigvec = power_method(-M, start, 10)
    x = chebyshev_approx_1d(n, beta, eigvec, L)
    res = {}
    res["x"], res["energy"], res["size"] = sweep_opt(x, F, G, k, ind)

    return res


def fast_search(G, F, k, n, ind):
    """
        Efficient version of cut computation.
        Does not perform 1-D search for beta.
        Input:
            * G: graph
            * F: graph signal
            * k: max edges to be cut
            * n: number of chebyshev polynomials
            * ind: vertex index vertex: unique integer
        Output:
            * cut
    """
    start = np.ones(nx.number_of_nodes(G))
    CAC = fast_cac(G, F, ind)

    return cheb_spectral_cut(CAC, start, F, G, 1., k, n, ind)


# optimal_wavelet_basis() builds the wavelet basis tree. To do so,
# it calls either the SWT (one_d_search()) or the FSWT (fast_search())


def optimal_wavelet_basis(G, F, k, npol, method='lobpcg'):
    """
        Computation of optimal graph wavelet basis.
        Input:
            * G: graph
            * F: graph signal
            * k: max edges to be cut
            * npol: number of chebyshev polynomials, if 0 run exact version
            * method: method used for Fiedler vector computation.
                The default value is 'lobpcg' however 'tracemin_lu' seems
                faster and it appears to give more stable results when used
                with PYTHONHASHSEED set to a constant value.
        Output:
            * root: tree root
            * ind: vertex index vertex: unique integer
            * size: number of edges cut
    """

    gsp.set_fiedler_method(method)

    # Creating index
    ind = {v: i for i, v in enumerate(G.nodes())}

    # First cut
    root = Node(None)
    size = 0
    cand_cuts = []

    if npol == 0:
        c = one_d_search(G, F, k, ind)
    else:
        c = fast_search(G, F, k, npol, ind)

    c["parent"] = root
    c["graph"] = G

    cand_cuts.append(c)
    # Recursively find the best cut. Each time it tries first the supposedly
    # biggest subgraphs inserted earlier in cand_cuts until the list is emptied
    # or no new best cut is found
    while size <= k and len(cand_cuts) > 0:
        best_cut = None
        for i, c in enumerate(cand_cuts):
            if (c["size"] + size <= k and c["energy"] > 0 and
                    (best_cut is None or c["energy"] > best_cut["energy"])):
                best_cut = c
                b = i

        # Exit iteration if no better cut is found
        if best_cut is None:
            break
        else:
            # Compute cut on left and right side
            G1, G2 = gsp.get_subgraphs(best_cut["graph"], best_cut["x"])

            size += best_cut["size"]

            for Gi in (G1, G2):
                if nx.number_of_nodes(Gi) == 1:
                    best_cut["parent"].add_child(
                        Node(ind[list(Gi.nodes())[0]]))
                elif nx.number_of_nodes(Gi) > 1:
                    n = Node(None)

                    if npol == 0:
                        c = one_d_search(Gi, F, k, ind)
                    else:
                        c = fast_search(Gi, F, k, npol, ind)

                    c["parent"] = n
                    c["graph"] = Gi
                    cand_cuts.append(c)

                    best_cut["parent"].add_child(n)

            del cand_cuts[b]

    # Compute remaining cuts using ratio cuts once the budget k is over
    # until the tree is completely built. (Not optimal search)
    for i in range(0, len(cand_cuts)):
        gsp.rc_recursive(cand_cuts[i]["parent"], cand_cuts[i]["graph"], ind)

    gsp.set_counts(root)

    return root, ind, size
