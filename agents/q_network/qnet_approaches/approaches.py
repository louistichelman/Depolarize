# qnets/approaches.py
from .complex_qnet import ComplexQNetwork
from .simple_qnet import SimpleQNetwork

QNET_REGISTRY = {
    "complex": ComplexQNetwork,
    "simple": SimpleQNetwork,
}
