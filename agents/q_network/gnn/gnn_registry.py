from .architectures.global_gnn import GlobalGNN
from .architectures.graph_sage import GraphSAGE
from .architectures.graphormer import Graphormer

GNN_REGISTRY = {"Global": GlobalGNN, "GraphSage": GraphSAGE, "Graphormer": Graphormer}
