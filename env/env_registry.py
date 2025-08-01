from .environments.fj_env import FJOpinionDynamics
from .environments.fj_env_simple import FJOpinionDynamicsFinite
from .environments.nonlinear_env import NLOpinionDynamics
from .base_env import BaseEnv

ENVIRONMENT_REGISTRY = {
    "nonlinear": NLOpinionDynamics,
    "friedkin_johnson": FJOpinionDynamics,
    "friedkin_johnson_simple": FJOpinionDynamicsFinite,
}
