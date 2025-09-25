from .architectures.global_mp import GlobalMP
from .architectures.graph_sage import GraphSAGE
from .architectures.graphormer_gd import GraphormerGD
from .architectures.gcn import GCN

GNN_REGISTRY = {
    "GlobalMP": GlobalMP,
    "GraphSage": GraphSAGE,
    "GraphormerGD": GraphormerGD,
    "GCN": GCN,
}
