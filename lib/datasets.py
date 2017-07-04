'''
The directories in data are structured as follows:
    datasetname-|
                |-datasetname.data
                |-datasetname.graph

   "datasetname.data" has info about the graph signal.
    Each row has format:
        "vertex_id,vertex_value"

    "datasetname.graph" has info about edges.
    If they are weighted each row has format:
        "vertx_A,vertex_B,edge_weight"
    Otherwise:
        "vertex_A,vertex_B"
'''


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
