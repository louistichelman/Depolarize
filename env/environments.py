from .fj_env import FJDepolarize
from .fj_env_simple import DepolarizeSimple
from .nonlinear_env import NLOpinionDynamics

ENVIRONMENT_REGISTRY = {
    "nonlinear": NLOpinionDynamics,
    "friedkin_johnson": FJDepolarize,
    "friedkin_johnson_simple": DepolarizeSimple
}