"""Context set construction for CARL environments.

The proposal uses one main testbed (CartPole), with a second environment
(LunarLander) only as a robustness check if time permits. Everything else
here (Pendulum, Acrobot, MountainCar, MountainCarContinuous, BipedalWalker) is
a bonus environment, not required by the proposal, added for broader
robustness-check coverage of the handling policies beyond what's strictly
asked for.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from carl.envs import CARLAcrobot, CARLCartPole, CARLMountainCar, CARLMountainCarContinuous, CARLPendulum

try:
    from carl.envs import CARLLunarLander
except ImportError:  # box2d extra not installed (e.g. native Windows, see README)
    CARLLunarLander = None

try:
    from carl.envs import CARLBipedalWalker
except ImportError:  # same box2d dependency as CARLLunarLander
    CARLBipedalWalker = None

ContextSet = dict[int, dict[str, float]]


@dataclass(frozen=True)
class EnvSpec:
    name: str
    carl_cls: type
    context_features: list[str]
    context_range: dict[str, tuple[float, float]]


ENV_SPECS: dict[str, EnvSpec] = {
    # Main testbed.
    "cartpole": EnvSpec(
        name="cartpole",
        carl_cls=CARLCartPole,
        context_features=["length", "masspole"],
        context_range={"length": (0.2, 2.2), "masspole": (0.05, 0.9)},
    ),
    # Optional second environment, robustness check only.
    # Needs the `box2d` extra (pip install "carl-bench[box2d]"), which needs a working
    # SWIG + C++ toolchain to build box2d-py - works on Linux/WSL, not on native Windows
    # unless a modern prebuilt `Box2D` wheel happens to already be installed (see README).
    # CARLLunarLander is None otherwise and using "lunarlander" will error.
    "lunarlander": EnvSpec(
        name="lunarlander",
        carl_cls=CARLLunarLander,
        context_features=["GRAVITY_Y", "MAIN_ENGINE_POWER"],
        context_range={"GRAVITY_Y": (-20.0, -6.0), "MAIN_ENGINE_POWER": (3.0, 18.0)},
    ),
    # Everything below is a bonus environment, not required by the proposal -
    # added for broader robustness-check coverage.
    "pendulum": EnvSpec(
        name="pendulum",
        carl_cls=CARLPendulum,
        context_features=["m", "l"],
        context_range={"m": (0.5, 3.5), "l": (0.5, 3.5)},
    ),
    # Discrete control, sparse reward (-1/step until swing-up) - a different
    # difficulty profile than CartPole/Pendulum's dense rewards.
    "acrobot": EnvSpec(
        name="acrobot",
        carl_cls=CARLAcrobot,
        context_features=["LINK_LENGTH_1", "LINK_MASS_1"],
        context_range={"LINK_LENGTH_1": (0.5, 3.0), "LINK_MASS_1": (0.5, 3.0)},
    ),
    # Notoriously hard for vanilla PPO without reward shaping (sparse, only
    # mildly informative reward) - included for diversity/robustness anyway;
    # don't be surprised if every policy struggles roughly equally here.
    "mountaincar": EnvSpec(
        name="mountaincar",
        carl_cls=CARLMountainCar,
        context_features=["force", "gravity"],
        context_range={"force": (0.0007, 0.002), "gravity": (0.001, 0.004)},
    ),
    "mountaincarcontinuous": EnvSpec(
        name="mountaincarcontinuous",
        carl_cls=CARLMountainCarContinuous,
        context_features=["power", "goal_position"],
        context_range={"power": (0.0008, 0.003), "goal_position": (0.4, 0.55)},
    ),
    # Needs the `box2d` extra, same as lunarlander (see README). Continuous
    # 4D action, richer/harder locomotion task than LunarLander.
    "bipedalwalker": EnvSpec(
        name="bipedalwalker",
        carl_cls=CARLBipedalWalker,
        context_features=["MOTORS_TORQUE", "FRICTION"],
        context_range={"MOTORS_TORQUE": (40.0, 120.0), "FRICTION": (0.5, 4.0)},
    ),
}


def get_env_spec(env_name: str) -> EnvSpec:
    if env_name not in ENV_SPECS:
        raise KeyError(f"Unknown env '{env_name}'. Known envs: {list(ENV_SPECS)}")
    return ENV_SPECS[env_name]


def build_context_set(env_name: str, seed: int, n_contexts: int = 50) -> ContextSet:
    """Sample `n_contexts` contexts by varying the env's context features uniformly."""
    spec = get_env_spec(env_name)
    rng = np.random.default_rng(seed)
    default = spec.carl_cls.get_default_context()

    contexts: ContextSet = {}
    for i in range(n_contexts):
        ctx = dict(default)
        for feat in spec.context_features:
            lo, hi = spec.context_range[feat]
            ctx[feat] = float(rng.uniform(lo, hi))
        contexts[i] = ctx
    return contexts
