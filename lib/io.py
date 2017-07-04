import networkx
import numpy as np
import pandas as pd
from datetime import datetime, date, time
import statsmodels.api as sm


def read_graph(input_graph_name, input_data_name):
    """
        Reads graph from file.
        Input:
            * input_graph_name: csv edge list
            * input_data_name: csv node-value pairs
        Output:
            * networkx graph: largest connected component
    """

    # Reading input data: graph signal values
    input_data = open(input_data_name, 'r')
    values = {}

    for line in input_data:
        line = line.rstrip()
        vec = line.rsplit(',')

        vertex = vec[0]
        value = float(vec[1])
        values[vertex] = value

    input_data.close()

    # Reading graph data: graph's edges
    G = networkx.Graph()
    input_graph = open(input_graph_name, 'r')

    for line in input_graph:
        line = line.rstrip()
        vec = line.rsplit(',')
        v1 = vec[0]
        v2 = vec[1]
        # The weighted case is not handled yet
        if v1 in values and v2 in values:
            G.add_edge(v1, v2, weight=1.)

    # Extracting largest connected component from graph
    Gcc = sorted(networkx.connected_component_subgraphs(G), key=len, reverse=True)

    G = Gcc[0]

    values_in_graph = {}

    # Setting values as node attributes
    for v in values.keys():
        if v in G:
            values_in_graph[v] = values[v]

    input_graph.close()
    networkx.set_node_attributes(G, "value", values_in_graph)

    return G


def read_values(input_data_name, G):
    """
        Reads node values.
        Input:
            * input_data_name: csv node-value pairs
            * G: networkx graph
        Output:
            * F: normalized node values array, ordered by G.nodes()
    """
    D = {}
    input_data = open(input_data_name, 'r')

    # Reading file
    for line in input_data:
        line = line.rstrip()
        vec = line.rsplit(',')

        vertex = vec[0]
        value = float(vec[1])
        D[vertex] = value

    input_data.close()

    F = []
    for v in G.nodes():
        if v in D:
            F.append(float(D[v]))
        else:
            F.append(0.)

    # Normalization
    F = np.array(F)
    F = F / np.max(F)
    F = F - np.mean(F)

    return F


def read_dyn_graph(path, num_snapshots, G):
    """
        Reads a dynamic graph.
        Input:
            * path: Path containing a files for each graph snapshot
                (e.g. folder/traffic_, for files folder/traffic_0.data ...
                folder/traffic_100.data)
            * num_snapshots: number of snapshots
            * G: networkx graph
        Output:
                * FT: array #snapshots x #vertices

    """
    FT = []
    for t in range(num_snapshots):
        in_file = path + "_" + str(t) + ".data"
        F = read_values(in_file, G)
        FT.append(F)

    return np.array(FT)


def clean_traffic_data(FT):
    # start_time = datetime.strptime("1/04/11 00:00", "%d/%m/%y %H:%M")
    c_FT = []
    for i in range(FT.shape[1]):
        # removing daily seasonality
        data = pd.DataFrame(FT[:, i], pd.DatetimeIndex(
            start='1/04/11 00:00', periods=len(FT[:, i]), freq='5min'))
        data.interpolate(inplace=True)

        res = sm.tsa.seasonal_decompose(data.values, freq=288)
        F = FT[:, i] - res.seasonal

        # removing weekly seasonality
        data = pd.DataFrame(F, pd.DatetimeIndex(
            start='1/04/11 00:00', periods=len(FT[:, i]), freq='5min'))
        res = sm.tsa.seasonal_decompose(data.values, freq=288 * 7)
        F = F - res.seasonal

        c_FT.append(F)

    return np.array(c_FT).transpose()
