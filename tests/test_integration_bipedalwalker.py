"""End-to-end smoke test with real CARLBipedalWalker. Skipped cleanly if a
working Box2D binding isn't available - same dependency as CARLLunarLander,
see README.
"""
import pytest

carl = pytest.importorskip("carl")

from src.envs.contexts import get_env_spec  # noqa: E402

if get_env_spec("bipedalwalker").carl_cls is None:
    pytest.skip("Box2D not available (CARLBipedalWalker unavailable)", allow_module_level=True)

from src.curriculum.controller import CurriculumController  # noqa: E402
from src.envs.contexts import build_context_set  # noqa: E402
from src.envs.factory import make_single_env, make_vec_env  # noqa: E402


def test_bipedalwalker_single_env_step_and_obs_shape():
    contexts = build_context_set("bipedalwalker", seed=0, n_contexts=4)
    controller = CurriculumController(context_ids=list(contexts.keys()), handling_policy="ignore")
    env = make_single_env("bipedalwalker", contexts, controller, seed=0)

    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert "context_id" in info


def test_bipedalwalker_obs_includes_context_features():
    contexts = build_context_set("bipedalwalker", seed=0, n_contexts=4)
    controller = CurriculumController(context_ids=list(contexts.keys()), handling_policy="ignore")
    env = make_single_env("bipedalwalker", contexts, controller, seed=0)
    # BipedalWalker's raw obs is 24-dim, plus 2 context features
    assert env.observation_space.shape[0] == 26


def test_bipedalwalker_vec_env_smoke():
    contexts = build_context_set("bipedalwalker", seed=0, n_contexts=4)
    controller = CurriculumController(context_ids=list(contexts.keys()), handling_policy="ignore")
    vec_env = make_vec_env("bipedalwalker", contexts, controller, n_envs=2, seed=0)
    obs = vec_env.reset()
    assert obs.shape[0] == 2
    actions = [vec_env.action_space.sample() for _ in range(2)]
    obs, rewards, dones, infos = vec_env.step(actions)
    assert obs.shape[0] == 2
    vec_env.close()
