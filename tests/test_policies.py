import numpy as np

from src.curriculum.policies import (
    DownweightRevisitPolicy,
    HardExcludePolicy,
    IgnorePolicy,
    ReplayPriorityPolicy,
    build_policy,
)
from src.curriculum.signal import DifficultySignal


def _signal_with_hard_context(hard_idx=0, n=3):
    signal = DifficultySignal(context_ids=list(range(n)))
    signal.seen[:] = True
    signal.ema_signal[:] = 1.0
    signal.ema_signal[hard_idx] = 5.0
    signal.is_hard[:] = False
    signal.is_hard[hard_idx] = True
    return signal


def test_ignore_policy_always_uniform_regardless_of_signal():
    policy = IgnorePolicy()
    signal = _signal_with_hard_context()
    weights = policy.compute_weights(signal)
    assert np.allclose(weights, 1 / 3)


def test_hard_exclude_suppresses_hard_context_but_keeps_it_reachable():
    policy = HardExcludePolicy(floor=0.02)
    signal = _signal_with_hard_context(hard_idx=0)
    weights = policy.compute_weights(signal)
    assert weights[0] < weights[1]
    assert weights[0] < weights[2]
    assert weights[0] > 0
    assert np.isclose(weights[1], weights[2])
    assert np.isclose(weights.sum(), 1.0)


def test_hard_exclude_never_auto_resets():
    policy = HardExcludePolicy(floor=0.02)
    signal = _signal_with_hard_context(hard_idx=0)
    w1 = policy.compute_weights(signal)
    for _ in range(20):
        w = policy.compute_weights(signal)
    assert np.isclose(w[0], w1[0])  # stays suppressed as long as signal.is_hard stays True


def test_downweight_revisit_suppresses_then_resets_after_cooldown():
    policy = DownweightRevisitPolicy(downweight_factor=0.3, cooldown=3)
    signal = _signal_with_hard_context(hard_idx=0)

    w_first = policy.compute_weights(signal)
    assert w_first[0] < w_first[1]

    # still within cooldown window
    for _ in range(2):
        w = policy.compute_weights(signal)
        assert w[0] < w[1]

    # cooldown elapsed -> forced revisit resets to full weight for one round
    w_reset = policy.compute_weights(signal)
    assert np.isclose(w_reset[0], w_reset[1])


def test_replay_priority_weights_track_continuous_signal_not_hard_flag():
    policy = ReplayPriorityPolicy(temperature=1.0, uniform_mix=0.0)
    signal = _signal_with_hard_context(hard_idx=0)
    weights = policy.compute_weights(signal)
    # context 0 has the highest raw signal (5.0 vs 1.0) -> highest weight,
    # purely because of signal magnitude, not the is_hard flag
    assert weights[0] > weights[1]
    assert np.isclose(weights[1], weights[2])
    assert np.isclose(weights.sum(), 1.0)


def test_replay_priority_uniform_mix_pulls_toward_uniform():
    signal = _signal_with_hard_context(hard_idx=0)
    sharp = ReplayPriorityPolicy(temperature=1.0, uniform_mix=0.0).compute_weights(signal)
    blended = ReplayPriorityPolicy(temperature=1.0, uniform_mix=0.9).compute_weights(signal)
    # more uniform_mix -> weights closer to 1/3 each
    assert abs(blended[0] - 1 / 3) < abs(sharp[0] - 1 / 3)


def test_build_policy_dispatches_correctly():
    assert build_policy("ignore").name == "ignore"
    assert build_policy("hard_exclude", floor=0.1).floor == 0.1
    assert build_policy("downweight_revisit", downweight_factor=0.5, cooldown=4).downweight_factor == 0.5
    assert build_policy("replay_priority", temperature=2.0).temperature == 2.0
