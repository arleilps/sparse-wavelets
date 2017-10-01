import os

import numpy as np
import networkx as nx
import scipy

import lib.optimal_cut as oc


def add_signal_to_graph(G, F, identifiers=None):
    r"""
        Assign the signal to the graph's nodes.
        Note that G.node[v] returns a dict.
        Input:
        * G: networkx graph
        * F: graph signal
        * identifiers: list of node identifiers. It's used to guarantee that
            the graph signal values are assigned to the right nodes
    """
    if identifiers is None:
        for i, v in enumerate(G.nodes()):
            G.node[v]["value"] = F[i]
    else:
        for i, v in enumerate(identifiers):
            G.node[v]["value"] = F[i]


def get_signal_from_graph(G):
    r"""
        Get the signal along with the list of nodes identifiers
    """
    F = []
    identifiers = []
    for v in G.nodes():
        identifiers.append(v)
        F.append(G.node[v]["value"])

    return np.array(F), identifiers


def rgb_to_hex(r, g, b):
    r"""
        Convert integer numbers to hexadecimal format.
        Input range for r, g, b: [0 - 255]
        Examples:
            black: (0, 0, 0)       -> #000000
            white: (255, 255, 255) -> #ffffff
    """
    return '#%02x%02x%02x' % (r, g, b)


def rgb(minimum, maximum, value):
    r"""
        Map 'value', from the range [minimum - maximum], to an RGB tuple. The
        output color scale covers a subset of the RGB spectrum
        Examples:
            value = maximum                      -> red     : #ff0000
            value = 0.5 * (maximum - minimum)    -> green   : #00ff00
            value = minimum                      -> blue    : #0000ff
    """
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value - minimum) / (maximum - minimum)
    b = int(max(0, 255 * (1 - ratio)))
    r = int(max(0, 255 * (ratio - 1)))
    g = 255 - b - r

    return rgb_to_hex(r, g, b)


def quote(s):
    return '"' + str(s) + '"'


def graph_to_dot(G, dot_output_file_name, nodes_color=(0, 255, 0)):
    r"""
        Write the graph description in .dot format to a file for visualization
        The graph doesn't need to have an attached signal
        Input:
            * G : networkx graph
            * dot_output_file_name : name of the output file (.dot format)
            * nodes_color : tuple of RGB coordinates (default is green)
    """
    output_file = open(dot_output_file_name, 'w')
    output_file.write('graph G{\n')
    output_file.write('rankdir="LR";\n')
    output_file.write('size=\"10,2\";\n')

    for v in G.nodes():
        color = rgb_to_hex(*nodes_color)
        output_file.write(quote(v) + '[shape=circle, label="", '
                          'style=filled, fillcolor=' + quote(color) + ', '
                          'penwidth=2, fixedsize=true, width=1, height=1]; \n')

    for edge in G.edges():
        output_file.write(quote(edge[0]) + ' -- ' + quote(edge[1]) +
                          ' [penwidth=1];\n')

    output_file.write("}")

    output_file.close()


def graph_with_values_to_dot(G, dot_output_file_name, maximum=None,
                             minimum=None, draw_zero_valued_nodes=False):
    r"""
        Write the graph description in .dot format to a file for visualization
        The graph needs to have an attached signal
        Input:
            * G : networkx graph with values (attached graph signal)
            * dot_output_file_name : name of the output file (.dot format)
            * minimum, maximum : are passed as arguments to the rgb() function
            * draw_zero_valued_nodes : if False, set penwidth = 0 for nodes
                with value 0
    """

    output_file = open(dot_output_file_name, 'w')
    output_file.write("graph G{\n")
    output_file.write("rankdir=LR;\n")
    output_file.write("size=\"10,2\";\n")

    if maximum is None:
        maximum = max([G.node[v]["value"] for v in list(G.nodes())])
    if minimum is None:
        minimum = min([G.node[v]["value"] for v in list(G.nodes())])

    for v in G.nodes():
        color = rgb(minimum, maximum, G.node[v]["value"])

        msg = (quote(v) + ' [shape=circle, label="", ' +
               'style=filled, fillcolor=' + quote(color) + ', ' +
               'penwidth={}, fixedsize=true, ' +
               'width=1, height=1]; \n')

        if G.node[v]["value"] != 0.0 or draw_zero_valued_nodes:
            output_file.write(msg.format(2))
        else:
            output_file.write(msg.format(0))

    for edge in G.edges():
        output_file.write(quote(edge[0]) + ' -- ' + quote(edge[1]) + ' ' +
                          '[penwidth=1];\n')

    output_file.write("}")

    output_file.close()


def dyn_graph_with_values_to_svg(G, FT, fig_output_file_name,
                                 path_to_svg_stack, fixed_color_scale=False,
                                 maximum=None, minimum=None,
                                 draw_zero_valued_nodes=False):
    r"""
        Create an .svg file of a dynamic graph. It stacks the various
        graph snapshots vertically. The graph signal change with time
        Input:
            * G : networkx graph
            * FT : temporal graph signal
            * fig_output_file_name : name of the output file (.svg format)
            * path_to_svg_stack : path to svg_stack.py, clone it from
                https://github.com/astraw/svg_stack
            * fixed_color_scale : if True the same maximum and minimum are used
                for every intermediate figure
            * minimum, maximum : are passed as arguments to the rgb() function.
                Not used if fixed_color_scale is set to False
            * draw_zero_valued_nodes : if False, set penwidth = 0 for nodes
                with value 0
    """

    if fixed_color_scale:
        if maximum is None:
            maximum = max([max(FT[i]) for i in range(FT.shape[0])])
        if minimum is None:
            minimum = min([min(FT[i]) for i in range(FT.shape[0])])

    svg_names = ''

    for i in range(FT.shape[0]):
        add_signal_to_graph(G, FT[i])

        dot_file_name = "dyn_graph-" + str(i) + ".dot"
        svg_file_name = "dyn_graph-" + str(i) + ".svg"

        graph_with_values_to_dot(G, dot_file_name, maximum, minimum,
                                 draw_zero_valued_nodes)

        # Use Scalable Force Directed Placement (sfdp), a fast multilevel force
        # directed algorithm to layout very large graphs with high quality.
        # It is available as part of the graphviz software.
        os.system('sfdp - Goverlap=prism - Tsvg ' + dot_file_name + ' > ' +
                  svg_file_name + ' ; rm ' + dot_file_name)
        svg_names += ' ' + svg_file_name

    # Stack all the previously created svg images vertically with no margin
    # in between the figures
    os.system("python " + path_to_svg_stack + " --direction=v --margin=0 " +
              svg_names + " > " + fig_output_file_name)

    for i in range(FT.shape[0]):
        os.system("rm dyn_graph-" + str(i) + ".svg")


def partitions_with_values_to_dot(G, partitions, dot_output_file_name,
                                  maximum=None, minimum=None,
                                  draw_zero_valued_nodes=False):
    r"""
        Create a .DOT file of a graph. The nodes within the same partition
        are connected by edges with thicker penwidth (4 vs 1).
        The graph doesn't need to have an attached signal
        Input:
            * G : networkx graph with values (attached graph signal)
            * dot_output_file_name : name of the output file (.dot format)
            * minimum, maximum : are passed as arguments to the rgb() function
            * draw_zero_valued_nodes : if False, set penwidth = 0 for nodes
                with value 0
    """
    output_file = open(dot_output_file_name, 'w')
    output_file.write('graph G{\n')
    output_file.write('rankdir=LR;\n')
    output_file.write('size="10,2";\n')

    if maximum is None:
        maximum = max([G.node[v]["value"] for v in G.nodes])
    if minimum is None:
        minimum = min([G.node[v]["value"] for v in G.nodes])

    part_map = {}

    for p in range(len(partitions)):
        for i in range(len(partitions[p])):
            part_map[partitions[p][i]] = p

    for v in G.nodes():
        color = rgb(minimum, maximum, G.node[v]["value"])
        msg = (quote(v) + '[shape=circle, label="", style=filled, fillcolor=' +
               quote(color) + ', penwidth={}, fixedsize=true, width=0.5, '
               'height=0.5]; \n')
        if G.node[v]["value"] != 0.0 or draw_zero_valued_nodes:
            output_file.write(msg.format(2))
        else:
            output_file.write(msg.format(0))

    for edge in G.edges():
        if part_map[edge[0]] == part_map[edge[1]]:
            output_file.write(quote(edge[0]) + ' -- ' + quote(edge[1]) +
                              ' [penwidth=4];\n')
        else:
            output_file.write(quote(edge[0]) + ' -- ' + quote(edge[1]) +
                              ' [penwidth=1];\n')

    output_file.write("}")

    output_file.close()


def time_graph_to_svg(G, fig_output_file_name, path_to_svg_stack,
                      nodes_color=(0, 255, 0)):
    r"""
        Create an .svg file of a dynamic graph. It stacks the various
        graph snapshots vertically. The edges change with time
        Input:
            * G : the graph has an associated list of snapshots (G.snap).
                Each of them is a networkx graph. They all have the same
                number of nodes and do not need to have an attached signal.
                The associated edges can vary
            * fig_output_file_name : name of the output file (.svg format)
            * path_to_svg_stack : path to svg_stack.py, clone it from
                https://github.com/astraw/svg_stack
            * nodes_color : tuple of RGB coordinates (default is green)
    """

    svg_names = ''

    for i in range(G.num_snaps()):
        snap_dot_file_name = "graph-" + str(i) + ".dot"
        snap_svg_file_name = "graph-" + str(i) + ".svg"
        graph_to_dot(G.snap(i), snap_dot_file_name, nodes_color)

        # Use Scalable Force Directed Placement (sfdp), a fast multilevel force
        # directed algorithm to layout very large graphs with high quality.
        # It is available as part of the graphviz software.
        os.system("sfdp -Goverlap=prism -Tsvg " + snap_dot_file_name +
                  "> " + snap_svg_file_name + " ; rm " + snap_dot_file_name)
        svg_names += " " + snap_svg_file_name

    # Stack all the previously created svg images vertically with no margin
    # in between the figures
    os.system("python " + path_to_svg_stack + " --direction=v --margin=0 " +
              svg_names + " > " + fig_output_file_name)

    for i in range(G.num_snaps()):
        os.system("rm graph-" + str(i) + ".svg")


def eig_vis_opt(G, F, beta):
    """
        Computes first and second eigenvector of sqrt(C+beta*L)^T CAC
        sqrt(C+beta*L) matrix for visualization.
        Input:
            * G: graph
            * F: graph signal
            * beta: regularization parameter
        Output:
            * v1: first eigenvector
            * v2: second eigenvector
    """

    ind = {v: i for i, v in enumerate(G.nodes())}

    C = oc.complete_graph_laplacian(nx.number_of_nodes(G))
    A = oc.weighted_adjacency_complete(G, F, ind)
    CAC = np.dot(np.dot(C, A), C)
    L = nx.laplacian_matrix(G).todense()

    isqrtCL = oc.sqrtmi(C + beta * L)
    M = np.dot(np.dot(isqrtCL, CAC), isqrtCL)

    (eigvals, eigvecs) = scipy.linalg.eigh(M, eigvals=(0, 1))
    v1 = np.asarray(np.dot(eigvecs[:, 0], isqrtCL))[0, :]
    v2 = np.asarray(np.dot(eigvecs[:, 1], isqrtCL))[0, :]

    return v1, v2
