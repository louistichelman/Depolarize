# qnets/approaches.py
from .architectures.complex_qnet import ComplexQNetwork
from .architectures.simple_qnet import SimpleQNetwork

QNET_REGISTRY = {
    "complex": ComplexQNetwork,
    "simple": SimpleQNetwork,
}
