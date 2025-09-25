# qnets/approaches.py
from .architectures.combining_embeddings_qnet import CE_QNetwork
from .architectures.direct_projection_qnet import DP_QNetwork

QNET_REGISTRY = {
    "CE": CE_QNetwork,
    "DP": DP_QNetwork,
}
