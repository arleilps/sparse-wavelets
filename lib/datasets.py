r"""

This module provides path to the dataset in data.
Each one is in a directory with the following structure:

    datasetname
      |- datasetname.data
      |- datasetname.graph

datasetname.data has info about the graph signal.
Row format : "vertex_id, vertex_value"

datasetname.graph has info about edges.
Row format: "vertex_A, vertex_B[, edge_weight]"

"""


# Small traffic (weighted)
# Speeds from traffic sensors
# Vertices: 100

small_traffic = {}
small_traffic["path"] = "data/small_traffic/"

# Large traffic (weigthed)
# Speeds from traffic sensors
# Vertices: 1923

traffic = {}
traffic["path"] = "data/traffic/"

# Human (unweighted)
# Gene expression data
# Vertices: 3628

human = {}
human["path"] = "data/human/"

# Wikipedia data (unweighted)
# Number of views of wikipedia pages
# Vertices: 4871

wiki = {}
wiki["path"] = "data/wiki/"

# Political blogs (unweighted)
# Link network of congressman's blogs with democrat/republican (0/1) as signal
# Vertices: 1490

polblogs = {}
polblogs["path"] = "data/polblogs/"
