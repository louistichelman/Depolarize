from .global_gnn import GlobalGNN
from .graph_sage import GraphSAGE
from .fj_graphormer import Graphormer

ARCHITECTURE_REGISTRY = {
    "global": GlobalGNN,
    "GraphSage": GraphSAGE,
    "Graphormer": Graphormer
}
