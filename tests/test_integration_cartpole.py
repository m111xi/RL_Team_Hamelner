"""End-to-end smoke test with real CARLCartPole: env creation and
curriculum-driven context selection. Skipped cleanly if carl-bench isn't
importable.
"""
import numpy as np
import pytest

carl = pytest.importorskip("carl")

from src.curriculum.controller import CurriculumController  # noqa: E402
from src.envs.contexts import build_context_set  # noqa: E402
from src.envs.factory import make_eval_env, make_single_env, make_vec_env  # noqa: E402


def test_build_context_set_has_correct_size():
    contexts = build_context_set("cartpole", seed=0, n_contexts=5)
    assert len(contexts) == 5


def test_single_env_step_and_obs_shape():
    contexts = build_context_set("cartpole", seed=0, n_contexts=4)
    controller = CurriculumController(context_ids=list(contexts.keys()), handling_policy="ignore")
    env = make_single_env("cartpole", contexts, controller, seed=0)

    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert "context_id" in info


def test_obs_includes_context_features():
    contexts = build_context_set("cartpole", seed=0, n_contexts=4)
    controller = CurriculumController(context_ids=list(contexts.keys()), handling_policy="ignore")
    env = make_single_env("cartpole", contexts, controller, seed=0)
    # CartPole's raw obs is 4-dim, plus 2 context features (length, masspole)
    assert env.observation_space.shape[0] == 6


def test_curriculum_controller_biases_context_sampling():
    contexts = build_context_set("cartpole", seed=0, n_contexts=3)
    context_ids = list(contexts.keys())
    controller = CurriculumController(context_ids=context_ids, handling_policy="replay_priority")
    # force weights to strongly favor a single context
    controller.weights = np.array([0.98, 0.01, 0.01])

    env = make_single_env("cartpole", contexts, controller, seed=0)
    seen = []
    for _ in range(20):
        _obs, info = env.reset()
        seen.append(info["context_id"])
    assert seen.count(context_ids[0]) > 10


def test_vec_env_smoke():
    contexts = build_context_set("cartpole", seed=0, n_contexts=4)
    controller = CurriculumController(context_ids=list(contexts.keys()), handling_policy="ignore")
    vec_env = make_vec_env("cartpole", contexts, controller, n_envs=2, seed=0)
    obs = vec_env.reset()
    assert obs.shape[0] == 2
    actions = [vec_env.action_space.sample() for _ in range(2)]
    obs, rewards, dones, infos = vec_env.step(actions)
    assert obs.shape[0] == 2
    vec_env.close()


def test_eval_env_always_uses_same_context():
    contexts = build_context_set("cartpole", seed=0, n_contexts=4)
    target_cid = list(contexts.keys())[0]
    env = make_eval_env("cartpole", contexts, target_cid, seed=0)
    for _ in range(5):
        _obs, info = env.reset()
        assert info["context_id"] == target_cid
