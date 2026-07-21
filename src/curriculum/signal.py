"""The one fixed difficulty signal shared by every handling policy.

Per the proposal, the project isolates the effect of *reacting* to hard
contexts, so the measurement of "is this context hard, and has it recovered"
must be identical across all compared policies. `DifficultySignal` owns that
measurement; `src/curriculum/policies.py` owns the reaction.
"""
from __future__ import annotations

import numpy as np


class DifficultySignal:
    def __init__(
        self,
        context_ids: list[int],
        ema_alpha: float = 0.3,
        hard_std_factor: float = 1.0,
        patience: int = 3,
    ):
        self.context_ids = list(context_ids)
        self.n = len(self.context_ids)
        self.ema_alpha = ema_alpha
        self.hard_std_factor = hard_std_factor
        self.patience = patience

        self._idx_of = {cid: i for i, cid in enumerate(self.context_ids)}
        self.ema_signal = np.zeros(self.n)
        self.ema_return = np.full(self.n, np.nan)
        self.seen = np.zeros(self.n, dtype=bool)
        self.is_hard = np.zeros(self.n, dtype=bool)
        self.hard_streak = np.zeros(self.n, dtype=int)
        self.recover_streak = np.zeros(self.n, dtype=int)

    def update(self, raw_signal: dict[int, float], returns: dict[int, float]) -> list[tuple[int, str]]:
        """Update EMAs from one rollout's data, return newly emitted
        (context_id, "flagged_hard" | "recovered") events."""
        for cid, value in raw_signal.items():
            idx = self._idx_of.get(cid)
            if idx is None:
                continue
            if not self.seen[idx]:
                self.ema_signal[idx] = value
                self.seen[idx] = True
            else:
                self.ema_signal[idx] = self.ema_alpha * value + (1 - self.ema_alpha) * self.ema_signal[idx]

        for cid, value in returns.items():
            idx = self._idx_of.get(cid)
            if idx is None:
                continue
            if np.isnan(self.ema_return[idx]):
                self.ema_return[idx] = value
            else:
                self.ema_return[idx] = self.ema_alpha * value + (1 - self.ema_alpha) * self.ema_return[idx]

        return self._update_hard_classification()

    def _update_hard_classification(self) -> list[tuple[int, str]]:
        seen_idx = np.where(~np.isnan(self.ema_return))[0]
        if len(seen_idx) >= 3:
            mean_r = float(np.mean(self.ema_return[seen_idx]))
            std_r = float(np.std(self.ema_return[seen_idx])) + 1e-8
            threshold = mean_r - self.hard_std_factor * std_r
        else:
            threshold = -np.inf

        events: list[tuple[int, str]] = []
        for idx in range(self.n):
            below = not np.isnan(self.ema_return[idx]) and self.ema_return[idx] < threshold

            if not self.is_hard[idx]:
                self.hard_streak[idx] = self.hard_streak[idx] + 1 if below else 0
                if self.hard_streak[idx] >= self.patience:
                    self.is_hard[idx] = True
                    self.recover_streak[idx] = 0
                    events.append((self.context_ids[idx], "flagged_hard"))
            else:
                self.recover_streak[idx] = self.recover_streak[idx] + 1 if not below else 0
                if self.recover_streak[idx] >= self.patience:
                    self.is_hard[idx] = False
                    self.hard_streak[idx] = 0
                    events.append((self.context_ids[idx], "recovered"))

        return events
