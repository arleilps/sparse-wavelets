import math
import time

import numpy as np
import matplotlib.pyplot as plt

import lib.optimal_cut as oc
import lib.syn as syn


def L2(F, F_approx):
    """
        Sum of squared errors
    """
    return sum([(F[i] - F_approx[i])**2 for i in range(F.shape[0])])


def size_time_experiment(sizes, balance, sparsity, energy, noise, num):
    """
        Size x time experiment using synthetic data.
        Input:
            * sizes: list of sizes
            * balance
            * sparsity
            * energy
            * noise
            * num: number of repetitions
        Output:
            * res_time: time results
    """
    res_time = []

    for s in range(len(sizes)):
        res_t = []
        for i in range(num):
            # synthetic_graph(size, num_edges, sparsity, energy, balance,
            # noise, seed=None)
            (G, F, k) = syn.synthetic_graph(sizes[s], 3 * sizes[s], sparsity,
                                            energy, balance, noise)

            ind = {v: i for i, v in enumerate(G.nodes())}

            start_time = time.time()
            oc.one_d_search(G, F, k, ind)
            time_slow = time.time() - start_time

            start_time = time.time()
            oc.fast_search(G, F, k, 5, ind)
            time_5 = time.time() - start_time

            start_time = time.time()
            oc.fast_search(G, F, k, 20, ind)
            time_20 = time.time() - start_time

            start_time = time.time()
            oc.fast_search(G, F, k, 50, ind)
            time_50 = time.time() - start_time

            res_t.append([time_slow, time_5, time_20, time_50])

        r = np.mean(np.array(res_t), axis=0)
        res_time.append(r)

    return np.array(res_time)


def sparsity_acc_experiment(sparsity, size, balance, energy, noise, num):
    """
        Sparsity x accuracy experiments using synthetic data.
        Input:
            * sparsity: many
            * size
            * balance
            * energy
            * noise
            * num: number of repetitions
        Output:
            * res: accuracy results
    """
    res = []

    for s in range(len(sparsity)):
        res_a = []
        for i in range(num):
            (G, F, k) = syn.synthetic_graph(size, 3 * size, sparsity[s],
                                            energy, balance, noise)

            ind = {v: i for i, v in enumerate(G.nodes())}

            c = oc.one_d_search(G, F, k, ind)
            acc_slow = c["energy"]

            c = oc.fast_search(G, F, k, 5, ind)
            acc_5 = c["energy"]

            c = oc.fast_search(G, F, k, 20, ind)
            acc_20 = c["energy"]

            c = oc.fast_search(G, F, k, 50, ind)
            acc_50 = c["energy"]

            res_a.append([acc_slow, acc_5, acc_20, acc_50])

        r = np.mean(np.array(res_a), axis=0)
        res.append(r)

    return np.array(res)


def noise_acc_experiment(noise, size, sparsity, energy, balance, num):
    """
        Noise x accuracy experiments using synthetic data.
        Input:
            * noise: list of noises
            * size
            * sparsity
            * energy
            * balance
            * num: number of repetitions
        Output:
            * res: accuracy results
    """
    res = []

    for s in range(len(noise)):
        res_a = []
        for i in range(num):
            (G, F, k) = syn.synthetic_graph(size, 3 * size, sparsity,
                                            energy, balance, noise[s])

            ind = {v: i for i, v in enumerate(G.nodes())}

            c = oc.one_d_search(G, F, k, ind)
            acc_slow = c["energy"]

            c = oc.fast_search(G, F, k, 5, ind)
            acc_5 = c["energy"]

            c = oc.fast_search(G, F, k, 20, ind)
            acc_20 = c["energy"]

            c = oc.fast_search(G, F, k, 50, ind)
            acc_50 = c["energy"]

            res_a.append([acc_slow, acc_5, acc_20, acc_50])

        r = np.mean(np.array(res_a), axis=0)
        res.append(r)

    return np.array(res)


def energy_acc_experiment(energy, size, sparsity, noise, balance, num):
    """
        Energy x accuracy experiments using synthetic data.
        Input:
            * energy: list of energies
            * size
            * sparsity
            * noise
            * balance
            * num: number of repetitions
        Output:
            * res: accuracy results
    """
    res = []

    for s in range(len(energy)):
        res_a = []
        for i in range(num):
            (G, F, k) = syn.synthetic_graph(size, 3 * size, sparsity,
                                            energy[s], balance, noise)

            ind = {v: i for i, v in enumerate(G.nodes())}

            c = oc.one_d_search(G, F, k, ind)
            acc_slow = c["energy"]

            c = oc.fast_search(G, F, k, 5, ind)
            acc_5 = c["energy"]

            c = oc.fast_search(G, F, k, 20, ind)
            acc_20 = c["energy"]

            c = oc.fast_search(G, F, k, 50, ind)
            acc_50 = c["energy"]

            res_a.append([acc_slow, acc_5, acc_20, acc_50])

        r = np.mean(np.array(res_a), axis=0)
        res.append(r)

    return np.array(res)


def plot_size_time_experiment(results, sizes, output_file_name):
    """
        Plots size x time experiment.
        Input:
            * results: time results
            * sizes: graph sizes
            * output_file_name: output file name
        Output:
            * None
    """
    plt.clf()

    ax = plt.subplot(111)

    ax.plot(sizes, results[:, 0], marker="x", color="cyan",
            label="SWT", markersize=15)
    ax.plot(sizes, results[:, 1], marker="o", color="orangered",
            label="FSWT-5", markersize=15)
    ax.plot(sizes, results[:, 2], marker="o", color="darkgreen",
            label="FSWT-20", markersize=15)
    ax.plot(sizes, results[:, 3], marker="o", color="k",
            label="FSWT-50", markersize=15)
    plt.gcf().subplots_adjust(bottom=0.15)
    ax.legend(loc='upper center', prop={'size': 20}, ncol=2)
    ax.set_ylabel('time (sec.)', fontsize=30)
    ax.set_xlabel('#vertices', fontsize=30)
    ax.tick_params(labelsize=23)
    # plt.rcParams['xtick.labelsize'] = 80
    # plt.rcParams['ytick.labelsize'] = 80
    ax.set_xlim([180, 1020])
    ax.set_ylim([0.01, 50000])
    ax.set_yscale('log')

    plt.savefig(output_file_name, dpi=300, bbox_inches='tight')


def plot_sparsity_acc_experiment(results, sparsity, output_file_name):
    """
        Plots sparsity x accuracy experiment.
        Input:
            * results: accuracy results
            * sparsity: sparsity values
            * output_file_name: output file name
        Output:
            * None
    """
    plt.clf()

    ax = plt.subplot(111)
    width = 0.04       # the width of the bars)
    ax.bar(np.array(sparsity) - 2 * width,
           results[:, 0], width, color='cyan', label="SWT", hatch="/")
    ax.bar(np.array(sparsity) - width, results[:, 1],
           width, color='orangered', label="FSWT-5", hatch="\\")
    ax.bar(np.array(sparsity), results[:, 2], width,
           color='darkgreen', label="FSWT-20", hatch="-")
    ax.bar(np.array(sparsity) + width, results[:, 3],
           width, color='k', label="FSWT-50", hatch="*")
    plt.gcf().subplots_adjust(bottom=0.15)
    ax.legend(loc='upper center', prop={'size': 20}, ncol=2)
    ax.set_ylabel(r'L$_2$ energy', fontsize=30)
    ax.set_xlabel('sparsity', fontsize=30)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    ax.set_xlim([0.1, 0.9])
    ax.set_ylim([0, 150])

    plt.savefig(output_file_name, dpi=300, bbox_inches='tight')


def plot_noise_acc_experiment(results, noise, output_file_name):
    """
        Plots noise x accuracy experiment.
        Input:
            * results: accuracy results
            * noise: noise values
            * output_file_name: output file name
        Output:
            * None
    """
    plt.clf()

    ax = plt.subplot(111)

    width = 0.04       # the width of the bars)
    ax.bar(np.array(noise) - 2 * width, results[:, 0],
           width, color='cyan', label="SWT")
    ax.bar(np.array(noise) - width,
           results[:, 1], width, color='orangered', label="FSWT-5")
    ax.bar(np.array(noise), results[:, 2],
           width, color='darkgreen', label="FSWT-20")
    ax.bar(np.array(noise) + width, results[:, 3],
           width, color='k', label="FSWT-50")

    plt.gcf().subplots_adjust(bottom=0.15)
    ax.legend(loc='upper center', prop={'size': 20}, ncol=2)
    ax.set_ylabel(r'L$_2$ energy', fontsize=30)
    ax.set_xlabel('noise', fontsize=30)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    ax.set_xlim([0.1, 0.9])
    ax.set_ylim([0, 150])

    plt.savefig(output_file_name, dpi=300, bbox_inches='tight')


def plot_energy_acc_experiment(results, energy, output_file_name):
    """
        Plots energy x accuracy experiment.
        Input:
            * results: accuracy results
            * energy: energy values
            * output_file_name: output file name
        Output:
            * None
    """
    plt.clf()

    ind = np.array(list(range(4)))
    ax = plt.subplot(111)
    width = 0.2      # the width of the bars)
    ax.bar(ind - width, results[:, 0],
           width, color='cyan', label="SWT", log=True)
    ax.bar(ind, results[:, 1],
           width, color='orangered', label="FSWT-5", log=True)
    ax.bar(ind + width, results[:, 2],
           width, color='darkgreen', label="FSWT-20", log=True)
    ax.bar(ind + 2 * width, results[:, 3],
           width, color='k', label="FSWT-50", log=True)

    plt.gcf().subplots_adjust(bottom=0.15)
    ax.legend(loc='upper left', prop={'size': 20}, ncol=2)
    ax.set_ylabel(r'L$_2$ energy', fontsize=30)
    ax.set_xlabel(r'L$_2$ energy (data)', fontsize=30)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    ax.set_yscale('log')
    ax.set_xticks(ind + width)
    ax.set_xticklabels((r'10$\mathregular{^1}$', r'10$\mathregular{^2}$',
                        r'10$\mathregular{^3}$', r'10$\mathregular{^4}$'))
    ax.set_ylim(0.1, 1000000)
    ax.set_xlim(-0.3, 3.7)

    plt.savefig(output_file_name, dpi=300, bbox_inches='tight')


def get_children(tree, chil_list):
    c = tree.children
    if c != []:
        chil_list.append(c[0])
        if len(c) == 2:
            chil_list.append(c[1])
        get_children(c[0], chil_list)
        if len(c) == 2:
            get_children(c[1], chil_list)


def compression_experiment(G, F, algs, comp_ratios, num):
    r"""
    Runs compression experiment static.
        Input:
            * G: graph
            * F: graph signal
            * algs: compression algorithms/transforms
            * comp_ratios: compression ratios
            * num: number of repetitions
        Output:
            * results: compression results
            * times: compression times

    The following DOCTEST is designed for NetworkX 1.11. Different versions
    (ex 2.0) return slightly different Fiedler vectors. Furthermore to get the
    same vector you should also set PYTHONHASHSEED=0. 'tracemin_lu' seems to
    be faster and give more stable results than 'lobpcg'. The resulting
    accuracy of the two method is similar.

    >>> from lib import io_utils, static
    >>> import lib.datasets as data
    >>>
    >>> G = io_utils.read_graph(data.small_traffic["path"] + "traffic.graph",
    ...                         data.small_traffic["path"] + "traffic.data")
    >>> F = io_utils.read_values(data.small_traffic["path"] +
    ...                         "traffic.data", G)
    >>> algs = [static.OptWavelets(n=5, method='lobpcg'),\
    ... static.OptWavelets(method='tracemin_lu'), static.Fourier(),\
    ... static.GRCWavelets(method='tracemin_lu'), static.HWavelets()]
    >>> comp_ratios = [0.1, 0.2]
    >>> res_smt, time_smt = \
    ...     compression_experiment(G, np.array(F), algs,
    ...                            comp_ratios, 1)
    >>> print(res_smt['FT'])
    [ 0.17681431  0.1036391 ]
    >>> print(res_smt['FSWT'])
    [ 0.16306688  0.03567735]
    >>> print(res_smt['SWT'])
    [ 0.15091373  0.05312383]
    >>> print(res_smt['GWT'])
    [ 0.1988195   0.10399963]
    >>> print(res_smt['HWT'])
    [ 0.27003969  0.23748869]
    """
    results = {}
    times = {}

    for alg in algs:
        results[alg.name()] = []
        times[alg.name()] = []
        for r in range(len(comp_ratios)):
            T = []
            R = []
            for i in range(num):
                start_time = time.time()
                alg.set_graph(G)
                tr = alg.transform(F)
                size = int(F.size * comp_ratios[r])
                appx_tr = alg.drop_frequency(tr, size)
                appx_F = alg.inverse(appx_tr)
                t = time.time() - start_time
                T.append(t)
                R.append(L2(F, appx_F))
            T = np.array(T)
            R = np.array(R)
            times[alg.name()].append(np.mean(T))
            results[alg.name()].append(np.mean(R))
        results[alg.name()] = np.array(results[alg.name()])
        times[alg.name()] = np.array(times[alg.name()])
    return results, times


def plot_compression_experiments(results, comp_ratios, output_file_name):
    """
        Plots compression size x accuracy experiment.
        Input:
            * results: accuracy results
            * comp_ratios: compression ratios
            * output_file_name: output file name
        Output:
            * None
    """

    # Get range of values
    highest_val = max([max(results[res]) for res in results])
    lowest_val = min([min(results[res]) for res in results])

    plt.clf()
    ax = plt.subplot(111)

    legend_columns = 3

    # Add some white space at the top to place the legend
    white_space = 0.3 * (len(results) // legend_columns
                         + bool((len(results) % legend_columns)))
    max_y = 10**(1 * math.log(highest_val, 10)
                 + white_space * (math.log(highest_val, 10)
                                  - math.log(lowest_val, 10)))

    ax.semilogy(comp_ratios, results["FSWT"], marker="o", color="r",
                label="FSWT")
    ax.semilogy(comp_ratios, results["FT"], marker="*", color="c",
                label="FT")
    if "SWT" in results:
        ax.semilogy(comp_ratios, results["SWT"], marker="x", color="b",
                    label="SWT")
    ax.semilogy(comp_ratios, results["GWT"], marker="s", color="g",
                label="GWT")
    if "HWT" in results:
        ax.semilogy(comp_ratios, results["HWT"], marker="v", color="y",
                    label="HWT")

    ax.legend(loc='upper center', ncol=legend_columns)
    ax.set_ylabel(r'L$_2$ error')
    ax.set_xlabel('size')

    ax.set_ylim(lowest_val, max_y)

    plt.savefig(output_file_name, dpi=300, bbox_inches='tight')
