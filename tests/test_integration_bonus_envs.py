"""End-to-end smoke tests for the bonus environments (not required by the
proposal, added for broader robustness-check coverage): Pendulum, Acrobot,
MountainCar, MountainCarContinuous. None of these need Box2D, so they run
wherever carl-bench itself is importable.
"""
import pytest

carl = pytest.importorskip("carl")

from src.curriculum.controller import CurriculumController  # noqa: E402
from src.envs.contexts import build_context_set, get_env_spec  # noqa: E402
from src.envs.factory import make_single_env, make_vec_env  # noqa: E402

BONUS_ENVS = ["pendulum", "acrobot", "mountaincar", "mountaincarcontinuous"]


@pytest.mark.parametrize("env_name", BONUS_ENVS)
def test_build_context_set_has_correct_size(env_name):
    contexts = build_context_set(env_name, seed=0, n_contexts=5)
    assert len(contexts) == 5


@pytest.mark.parametrize("env_name", BONUS_ENVS)
def test_single_env_step_and_obs_shape(env_name):
    contexts = build_context_set(env_name, seed=0, n_contexts=4)
    controller = CurriculumController(context_ids=list(contexts.keys()), handling_policy="ignore")
    env = make_single_env(env_name, contexts, controller, seed=0)

    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert "context_id" in info


@pytest.mark.parametrize("env_name", BONUS_ENVS)
def test_obs_includes_two_context_features(env_name):
    contexts = build_context_set(env_name, seed=0, n_contexts=4)
    spec = get_env_spec(env_name)
    controller = CurriculumController(context_ids=list(contexts.keys()), handling_policy="ignore")
    env = make_single_env(env_name, contexts, controller, seed=0)
    assert len(spec.context_features) == 2
    # flattened obs = raw obs dims + 2 context feature dims
    raw_env = spec.carl_cls()
    raw_obs_dim = raw_env.observation_space["obs"].shape[0]
    assert env.observation_space.shape[0] == raw_obs_dim + 2


@pytest.mark.parametrize("env_name", BONUS_ENVS)
def test_vec_env_smoke(env_name):
    contexts = build_context_set(env_name, seed=0, n_contexts=4)
    controller = CurriculumController(context_ids=list(contexts.keys()), handling_policy="ignore")
    vec_env = make_vec_env(env_name, contexts, controller, n_envs=2, seed=0)
    obs = vec_env.reset()
    assert obs.shape[0] == 2
    actions = [vec_env.action_space.sample() for _ in range(2)]
    obs, rewards, dones, infos = vec_env.step(actions)
    assert obs.shape[0] == 2
    vec_env.close()
