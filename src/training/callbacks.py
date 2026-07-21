"""SB3 callbacks that (1) turn PPO's own rollout data into the fixed difficulty
signal and (2) periodically evaluate every context deterministically."""
from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CurriculumUpdateCallback(BaseCallback):
    """After every PPO rollout, compute the error-based difficulty signal from
    the just-collected rollout buffer and push it into the shared
    CurriculumController, which updates the DifficultySignal and applies the
    configured handling policy. Logs every finished episode and every
    flagged-hard/recovered event.
    """

    def __init__(self, controller, run_logger, verbose: int = 0):
        super().__init__(verbose)
        self.controller = controller
        self.run_logger = run_logger
        self._context_ids_step: list[np.ndarray] = []
        self._episode_returns: dict[int, list[float]] = {}

    def _on_rollout_start(self) -> None:
        self._context_ids_step = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        ids = np.array([info.get("context_id", -1) for info in infos], dtype=np.int64)
        self._context_ids_step.append(ids)
        for info in infos:
            ep = info.get("episode")
            if ep is not None and ep.get("context_id") is not None:
                cid = int(ep["context_id"])
                self._episode_returns.setdefault(cid, []).append(float(ep["r"]))
                self.run_logger.log_episode(self.num_timesteps, cid, float(ep["r"]), int(ep["l"]))
        return True

    def _on_rollout_end(self) -> None:
        buffer = self.model.rollout_buffer
        values = buffer.values  # (n_steps, n_envs)
        returns = buffer.returns  # (n_steps, n_envs)
        context_ids = np.stack(self._context_ids_step, axis=0) if self._context_ids_step else np.zeros((0, 0))

        error = np.abs(values - returns)
        raw_signal = self._aggregate_per_context(context_ids, error)
        mean_returns = {cid: float(np.mean(rs)) for cid, rs in self._episode_returns.items() if rs}

        events = self.controller.update_from_signal(raw_signal, mean_returns)
        for cid, event in events:
            self.run_logger.log_hard_event(self.num_timesteps, cid, event)
        self.run_logger.log_curriculum_weights(self.num_timesteps, self.controller)
        self.run_logger.log_signal_state(self.num_timesteps, self.controller)

        self._episode_returns = {}

    @staticmethod
    def _aggregate_per_context(context_ids: np.ndarray, values_arr: np.ndarray) -> dict[int, float]:
        flat_ids = context_ids.reshape(-1)
        flat_vals = values_arr.reshape(-1)
        out: dict[int, float] = {}
        for cid in np.unique(flat_ids):
            if cid < 0:
                continue
            mask = flat_ids == cid
            out[int(cid)] = float(np.mean(flat_vals[mask]))
        return out


class PeriodicEvalCallback(BaseCallback):
    """Every `eval_freq` steps, run deterministic rollouts on every context in
    the context set - a clean, low-variance return signal distinct from the
    noisy on-policy training episodes."""

    def __init__(
        self,
        env_name: str,
        contexts: dict[int, dict[str, float]],
        eval_freq: int,
        episodes_per_context: int,
        run_logger,
        seed: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.env_name = env_name
        self.contexts = contexts
        self.eval_freq = eval_freq
        self.episodes_per_context = episodes_per_context
        self.run_logger = run_logger
        self.seed = seed
        self._last_eval = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval >= self.eval_freq:
            self._last_eval = self.num_timesteps
            self._run_eval()
        return True

    def _run_eval(self) -> None:
        from src.envs.factory import make_eval_env

        for cid in self.contexts:
            env = make_eval_env(self.env_name, self.contexts, cid, seed=self.seed)
            returns = []
            for ep in range(self.episodes_per_context):
                obs, _ = env.reset(seed=self.seed + ep)
                done = False
                ep_return = 0.0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _info = env.step(action)
                    ep_return += float(reward)
                    done = terminated or truncated
                returns.append(ep_return)
            env.close()
            self.run_logger.log_eval(self.num_timesteps, cid, float(np.mean(returns)))
