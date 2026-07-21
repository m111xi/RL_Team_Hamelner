"""Observation flattening, plus episode-stat tracking.

CARL's base observation is ``Dict({"obs": Box, "context": Box})`` (with
``obs_context_as_dict=False``). We turn this into a single flat ``Box`` by
concatenating obs and context, so plain SB3 MLP policies can be used and the
agent is given the context explicitly (matching the proposal's
:math:`\\pi_\\theta : S \\times C \\mapsto A`).
"""
from __future__ import annotations

import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class EpisodeStatsWrapper(gym.Wrapper):
    """Tags episode-end info with return/length/context_id.

    Uses the same ``info["episode"] = {"r", "l", "t"}`` convention as SB3's
    ``Monitor`` wrapper (so SB3's own ``rollout/ep_rew_mean`` logging keeps
    working), plus a ``context_id`` field the curriculum callback reads.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._ep_return = 0.0
        self._ep_len = 0
        self._start_time = time.time()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._ep_return = 0.0
        self._ep_len = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ep_return += float(reward)
        self._ep_len += 1
        if terminated or truncated:
            info = dict(info)
            info["episode"] = {
                "r": self._ep_return,
                "l": self._ep_len,
                "t": round(time.time() - self._start_time, 6),
                "context_id": info.get("context_id"),
            }
        return obs, reward, terminated, truncated, info


class FlattenContextObsWrapper(gym.ObservationWrapper):
    """Concatenate obs["obs"] and obs["context"] into a single flat Box."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_space = env.observation_space["obs"]
        ctx_space = env.observation_space["context"]
        low = np.concatenate([obs_space.low, ctx_space.low]).astype(np.float32)
        high = np.concatenate([obs_space.high, ctx_space.high]).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        obs = np.asarray(observation["obs"], dtype=np.float32)
        ctx = np.asarray(observation["context"], dtype=np.float32)
        return np.concatenate([obs, ctx])
