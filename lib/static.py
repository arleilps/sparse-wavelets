import math
import operator

import numpy as np
import networkx as nx

import lib.graph_signal_proc as gsp
import lib.optimal_cut as oc


class Fourier(object):
    """
        Graph Fourier transform.
    """

    def name(self):
        return "FT"

    def set_graph(self, _G):
        self.G = _G
        L = nx.laplacian_matrix(self.G)
        L = L.todense()
        self.U, self.lamb_str = gsp.compute_eigenvectors_and_eigenvalues(L)

    def transform(self, F):
        return gsp.graph_fourier(F, self.U)

    def inverse(self, ftr):
        return gsp.graph_fourier_inverse(ftr, self.U)

    def drop_frequency(self, ftr, n):
        """
            Keeps only the n top-energy coefficients of ftr.
            Input:
                * ftr: transform
                * n: number of coefficients
            Output:
                * ftr_copy: changed transform
        """
        coeffs = {i: abs(ftr[i]) for i in range(ftr.shape[0])}
        sorted_coeffs = sorted(coeffs.items(), key=operator.itemgetter(1),
                               reverse=True)
        ftr_copy = np.copy(ftr)
        # Set the other coefficients to 0
        for k in range(n, len(sorted_coeffs)):
            ftr_copy[sorted_coeffs[k][0]] = 0

        return ftr_copy


class HWavelets(object):
    """
        Hammond's wavelets (spectral theory)
    """

    def name(self):
        return "HWT"

    def set_graph(self, _G):
        """
        """
        self.G = _G
        L = nx.normalized_laplacian_matrix(self.G)
        L = L.todense()
        self.U, self.lamb_str = gsp.compute_eigenvectors_and_eigenvalues(L)
        lamb_max = max(self.lamb_str.real)

        # default parameters defined by author
        K = 100
        J = 4
        gamma = gsp.comp_gamma()
        self.T = gsp.comp_scales(lamb_max, K, J)
        self.w = gsp.graph_wavelets(self.lamb_str.real, self.U.real,
                                    len(self.G.nodes()), self.T)
        self.s = gsp.graph_low_pass(self.lamb_str.real, self.U.real,
                                    self.T, gamma, lamb_max, K)

    def transform(self, F):
        return gsp.hammond_wavelet_transform(self.w, self.s, self.T, F)

    def inverse(self, wtr):
        return gsp.hammond_wavelets_inverse(self.w, self.s, wtr)

    def drop_frequency(self, wtr, n):
        """
            Keeps only the n top-energy coefficients of wtr.
            Input:
                * wtr: transform
                * n: number of coefficients
            Output:
                * wtr_copy: changed transform
        """
        coeffs = {}
        for i in range(len(wtr)):
            for j in range(len(wtr[i])):
                coeffs[(i, j)] = abs(wtr[i][j])

        sorted_coeffs = sorted(coeffs.items(), key=operator.itemgetter(1),
                               reverse=True)

        wtr_copy = np.copy(wtr)

        for k in range(n, len(sorted_coeffs)):
            i = sorted_coeffs[k][0][0]
            j = sorted_coeffs[k][0][1]

            wtr_copy[i][j] = 0.

        return wtr_copy


class GRCWavelets(object):
    """
        Gavish's wavelet transform.
    """

    def __init__(self, method='lobpcg'):
        # Method for Fiedler vector computation
        self.method = method

    def name(self):
        return "GWT"

    def set_graph(self, G):
        self.G = G
        (self.tree, self.ind) = gsp.ratio_cut_hierarchy(self.G, self.method)

    def transform(self, F):
        return gsp.gavish_wavelet_transform(self.tree, self.G, F)

    def inverse(self, wtr):
        return gsp.gavish_wavelet_inverse(self.tree, self.ind, self.G, wtr)

    def drop_frequency(self, wtr, n):
        """
            Keeps only the n top-energy coefficients of wtr.
            Input:
                * wtr: transform
                * n: number of coefficients
            Output:
                * wtr_copy: changed transform
        """
        coeffs = {i: abs(wtr[i]) for i in range(len(wtr))}
        sorted_coeffs = sorted(coeffs.items(), key=operator.itemgetter(1),
                               reverse=True)

        wtr_copy = np.copy(wtr)

        for k in range(n, len(sorted_coeffs)):
            i = sorted_coeffs[k][0]

            wtr_copy[i] = 0.

        return wtr_copy


class OptWavelets(object):
    """
        Sparse wavelet transform.
    """

    def __init__(self, n=0, method='lobpcg'):
        self.n = n
        # Method for Fiedler vector computation
        self.method = method

    def name(self):
        if self.n == 0:
            return "SWT"
        else:
            return "FSWT"

    def set_graph(self, G):
        self.G = G

    def transform(self, F):
        self.F = F
        return None

    def inverse(self, wtr):
        return gsp.gavish_wavelet_inverse(self.tree, self.ind, self.G, wtr)

    def drop_frequency(self, wtr, n):
        # The number of edges to be cut is set equal to the number of
        # Chebyshev polynomials
        k = n
        # Computing optimal basis
        (self.tree, self.ind, size) = \
            oc.optimal_wavelet_basis(self.G, self.F, k,
                                     self.n, self.method)
        # Gavish's wavelet transform
        tr = gsp.gavish_wavelet_transform(self.tree, self.G, self.F)

        coeffs = {i: abs(tr[i]) for i in range(len(tr))}
        sorted_coeffs = sorted(coeffs.items(), key=operator.itemgetter(1),
                               reverse=True)

        wtr_copy = np.copy(tr)

        # Computing number of integers required to represent the
        # edges cut (rounded)
        v = n - int(math.ceil(float(size *
                                    math.log(len(self.G.edges()), 2)) / 64))

        for k in range(v, len(sorted_coeffs)):
            i = sorted_coeffs[k][0]

            wtr_copy[i] = 0.

        return wtr_copy
