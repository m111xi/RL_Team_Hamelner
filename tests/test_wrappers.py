import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.envs.wrappers import EpisodeStatsWrapper, FlattenContextObsWrapper


class DummyContextEnv(gym.Env):
    """Minimal CARL-shaped env: Dict({"obs": Box(4,), "context": Box(2,)})."""

    def __init__(self, episode_len: int = 3):
        super().__init__()
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                "context": spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(2)
        self.episode_len = episode_len
        self._t = 0
        self.context_id = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        obs = {"obs": np.zeros(4, dtype=np.float32), "context": np.array([1.0, 2.0], dtype=np.float32)}
        return obs, {"context_id": self.context_id}

    def step(self, action):
        self._t += 1
        obs = {"obs": np.full(4, self._t, dtype=np.float32), "context": np.array([1.0, 2.0], dtype=np.float32)}
        terminated = self._t >= self.episode_len
        return obs, 1.0, terminated, False, {"context_id": self.context_id}


def test_flatten_context_obs_wrapper_shape_and_values():
    env = FlattenContextObsWrapper(DummyContextEnv())
    assert env.observation_space.shape == (6,)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (6,)
    assert np.allclose(obs[:4], 0.0)
    assert np.allclose(obs[4:], [1.0, 2.0])


def test_episode_stats_wrapper_reports_return_and_length_and_context_id():
    env = EpisodeStatsWrapper(DummyContextEnv(episode_len=3))
    env.reset(seed=0)
    for _ in range(2):
        _obs, _r, terminated, _trunc, info = env.step(0)
        assert "episode" not in info
        assert not terminated
    _obs, _r, terminated, _trunc, info = env.step(0)
    assert terminated
    assert info["episode"]["r"] == 3.0
    assert info["episode"]["l"] == 3
    assert info["episode"]["context_id"] == 0


def test_wrappers_compose_with_episode_stats():
    env = FlattenContextObsWrapper(EpisodeStatsWrapper(DummyContextEnv(episode_len=2)))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (6,)
    _obs, _r, _term, _trunc, info = env.step(0)
    obs, _r, terminated, _trunc, info = env.step(0)
    assert terminated
    assert info["episode"]["r"] == 2.0
    assert obs.shape == (6,)
