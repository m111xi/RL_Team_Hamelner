"""Wires together a CARL env, the explicit-context observation wrapper, and the
curriculum context selector."""
from __future__ import annotations

from stable_baselines3.common.vec_env import DummyVecEnv

from src.curriculum.selector import build_curriculum_selector, build_fixed_context_selector
from src.envs.contexts import ContextSet, get_env_spec
from src.envs.wrappers import EpisodeStatsWrapper, FlattenContextObsWrapper


def make_single_env(env_name: str, contexts: ContextSet, controller, seed: int):
    """Build one training env whose context sampling is driven by `controller`."""
    spec = get_env_spec(env_name)
    selector = build_curriculum_selector(contexts, controller)
    env = spec.carl_cls(
        contexts=contexts,
        obs_context_features=spec.context_features,
        obs_context_as_dict=False,
        context_selector=selector,
    )
    env = EpisodeStatsWrapper(env)
    env = FlattenContextObsWrapper(env)
    env.reset(seed=seed)
    return env


def make_vec_env(env_name: str, contexts: ContextSet, controller, n_envs: int, seed: int):
    def _thunk(rank: int):
        def _init():
            return make_single_env(env_name, contexts, controller, seed + rank)

        return _init

    return DummyVecEnv([_thunk(i) for i in range(n_envs)])


def make_eval_env(env_name: str, contexts: ContextSet, context_id: int, seed: int):
    """Build an env that always resets into a single, fixed context (for eval)."""
    spec = get_env_spec(env_name)
    selector = build_fixed_context_selector(contexts, context_id)
    env = spec.carl_cls(
        contexts=contexts,
        obs_context_features=spec.context_features,
        obs_context_as_dict=False,
        context_selector=selector,
    )
    env = EpisodeStatsWrapper(env)
    env = FlattenContextObsWrapper(env)
    env.reset(seed=seed)
    return env
