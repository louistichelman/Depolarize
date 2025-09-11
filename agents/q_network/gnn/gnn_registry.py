from .architectures.global_gnn import GlobalGNN
from .architectures.graph_sage import GraphSAGE
from .architectures.graphormer import Graphormer
from .architectures.gcn import GCN

GNN_REGISTRY = {
    "Global": GlobalGNN,
    "GraphSage": GraphSAGE,
    "Graphormer": Graphormer,
    "GCN": GCN,
}
