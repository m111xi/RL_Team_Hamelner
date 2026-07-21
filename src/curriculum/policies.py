"""The four hard-context handling policies compared in the experiments.

All of them sit on top of the same uniform sampling backbone (per the
proposal's "Base Sampling Scheme") and read from the same shared
`DifficultySignal` (`src/curriculum/signal.py`) — the only thing that differs
between them is how they turn "this context is currently classified as hard"
(and, for replay-priority, the continuous signal magnitude) into sampling
weights.
"""
from __future__ import annotations

import numpy as np

from src.curriculum.signal import DifficultySignal

POLICY_NAMES = ("ignore", "hard_exclude", "downweight_revisit", "replay_priority")


class HandlingPolicy:
    name: str

    def compute_weights(self, signal: DifficultySignal) -> np.ndarray:
        raise NotImplementedError


class IgnorePolicy(HandlingPolicy):
    """Keep sampling uniformly; never react to the hard-context signal."""

    name = "ignore"

    def compute_weights(self, signal: DifficultySignal) -> np.ndarray:
        return np.ones(signal.n) / signal.n


class HardExcludePolicy(HandlingPolicy):
    """Temporarily pause contexts classified as too hard until their score improves.

    Not a literal zero: a paused context can only stop being "hard" if its
    return improves, which needs occasional sampling. Whether that trickle is
    enough for hard-excluded contexts to ever actually recover is itself part
    of what the experiment is meant to show.
    """

    name = "hard_exclude"

    def __init__(self, floor: float = 0.02):
        self.floor = floor

    def compute_weights(self, signal: DifficultySignal) -> np.ndarray:
        w = np.where(signal.is_hard, self.floor, 1.0)
        return w / w.sum()


class DownweightRevisitPolicy(HandlingPolicy):
    """Reduce the sampling probability of hard contexts, but keep revisiting them.

    Uniform backbone; hard contexts multiplied by `downweight_factor`. After
    `cooldown` rollouts of being downweighted, a context is reset to full
    weight for one round regardless of whether the signal says it improved -
    a forced, scheduled revisit rather than the purely signal-driven recovery
    that `hard_exclude` relies on.
    """

    name = "downweight_revisit"

    def __init__(self, downweight_factor: float = 0.3, cooldown: int = 8):
        self.downweight_factor = downweight_factor
        self.cooldown = cooldown
        self._cooldown_counter: np.ndarray | None = None
        self._suppressed: np.ndarray | None = None

    def compute_weights(self, signal: DifficultySignal) -> np.ndarray:
        if self._cooldown_counter is None:
            self._cooldown_counter = np.zeros(signal.n, dtype=int)
            self._suppressed = np.zeros(signal.n, dtype=bool)

        for idx in range(signal.n):
            if self._suppressed[idx]:
                self._cooldown_counter[idx] -= 1
                if self._cooldown_counter[idx] <= 0:
                    self._suppressed[idx] = False
            elif signal.is_hard[idx]:
                self._suppressed[idx] = True
                self._cooldown_counter[idx] = self.cooldown

        w = np.where(self._suppressed, self.downweight_factor, 1.0)
        return w / w.sum()


class ReplayPriorityPolicy(HandlingPolicy):
    """Keep hard contexts in the pool, adapt probability to learning potential.

    Blends the uniform backbone with a softmax over the continuous EMA signal
    (higher error/uncertainty -> higher priority) - unlike the other three
    policies, this never gates on the binary hard/not-hard classification, it
    reacts continuously to the raw signal magnitude, like PLR-style scoring.
    """

    name = "replay_priority"

    def __init__(self, temperature: float = 1.0, uniform_mix: float = 0.05):
        self.temperature = temperature
        self.uniform_mix = uniform_mix

    def compute_weights(self, signal: DifficultySignal) -> np.ndarray:
        raw = signal.ema_signal.copy()
        if not np.all(signal.seen) and signal.seen.any():
            raw[~signal.seen] = np.mean(raw[signal.seen])

        std = float(np.std(raw))
        z = (raw - float(np.mean(raw))) / (std + 1e-8) if std > 1e-8 else np.zeros_like(raw)

        exp = np.exp(z / max(self.temperature, 1e-6))
        softmax_w = exp / exp.sum()

        uniform = np.ones(signal.n) / signal.n
        w = (1 - self.uniform_mix) * softmax_w + self.uniform_mix * uniform
        return w / w.sum()


def build_policy(name: str, **kwargs) -> HandlingPolicy:
    if name == "ignore":
        return IgnorePolicy()
    if name == "hard_exclude":
        return HardExcludePolicy(floor=kwargs.get("floor", 0.02))
    if name == "downweight_revisit":
        return DownweightRevisitPolicy(
            downweight_factor=kwargs.get("downweight_factor", 0.3),
            cooldown=kwargs.get("cooldown", 8),
        )
    if name == "replay_priority":
        return ReplayPriorityPolicy(
            temperature=kwargs.get("temperature", 1.0),
            uniform_mix=kwargs.get("uniform_mix", 0.05),
        )
    raise ValueError(f"Unknown handling policy '{name}', expected one of {POLICY_NAMES}")
