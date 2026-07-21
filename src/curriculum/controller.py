"""Glues one DifficultySignal to one HandlingPolicy and exposes context-sampling weights."""
from __future__ import annotations

import numpy as np

from src.curriculum.policies import build_policy
from src.curriculum.signal import DifficultySignal


class CurriculumController:
    def __init__(
        self,
        context_ids: list[int],
        handling_policy: str,
        ema_alpha: float = 0.3,
        hard_std_factor: float = 1.0,
        patience: int = 3,
        **policy_kwargs,
    ):
        self.context_ids = list(context_ids)
        self.handling_policy_name = handling_policy
        self.signal = DifficultySignal(
            context_ids=self.context_ids,
            ema_alpha=ema_alpha,
            hard_std_factor=hard_std_factor,
            patience=patience,
        )
        self.policy = build_policy(handling_policy, **policy_kwargs)
        self.weights = np.ones(len(self.context_ids)) / len(self.context_ids)
        self.n_updates = 0
        self.weight_history: list[tuple[int, np.ndarray]] = []

    def get_weights(self) -> np.ndarray:
        return self.weights

    def update_from_signal(self, raw_signal: dict[int, float], returns: dict[int, float]) -> list[tuple[int, str]]:
        self.n_updates += 1
        events = self.signal.update(raw_signal, returns)
        weights = self.policy.compute_weights(self.signal)
        total = weights.sum()
        self.weights = weights / total if total > 1e-12 else np.ones(len(self.context_ids)) / len(self.context_ids)
        self.weight_history.append((self.n_updates, self.weights.copy()))
        return events
