"""End-to-end smoke test with real CARLLunarLander. Skipped cleanly if a
working Box2D binding isn't available - e.g. on native Windows without the
maintained `Box2D` wheel or WSL, see README.
"""
import pytest

carl = pytest.importorskip("carl")

from src.envs.contexts import get_env_spec  # noqa: E402

if get_env_spec("lunarlander").carl_cls is None:
    pytest.skip("Box2D not available (CARLLunarLander unavailable)", allow_module_level=True)

from src.curriculum.controller import CurriculumController  # noqa: E402
from src.envs.contexts import build_context_set  # noqa: E402
from src.envs.factory import make_single_env, make_vec_env  # noqa: E402


def test_lunarlander_single_env_step_and_obs_shape():
    contexts = build_context_set("lunarlander", seed=0, n_contexts=4)
    controller = CurriculumController(context_ids=list(contexts.keys()), handling_policy="ignore")
    env = make_single_env("lunarlander", contexts, controller, seed=0)

    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert "context_id" in info


def test_lunarlander_obs_includes_context_features():
    contexts = build_context_set("lunarlander", seed=0, n_contexts=4)
    controller = CurriculumController(context_ids=list(contexts.keys()), handling_policy="ignore")
    env = make_single_env("lunarlander", contexts, controller, seed=0)
    # LunarLander's raw obs is 8-dim, plus 2 context features (GRAVITY_Y, MAIN_ENGINE_POWER)
    assert env.observation_space.shape[0] == 10


def test_lunarlander_vec_env_smoke():
    contexts = build_context_set("lunarlander", seed=0, n_contexts=4)
    controller = CurriculumController(context_ids=list(contexts.keys()), handling_policy="ignore")
    vec_env = make_vec_env("lunarlander", contexts, controller, n_envs=2, seed=0)
    obs = vec_env.reset()
    assert obs.shape[0] == 2
    actions = [vec_env.action_space.sample() for _ in range(2)]
    obs, rewards, dones, infos = vec_env.step(actions)
    assert obs.shape[0] == 2
    vec_env.close()
