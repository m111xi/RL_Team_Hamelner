import numpy as np

from src.curriculum.signal import DifficultySignal


def test_ema_return_tracked_and_hard_not_flagged_when_uniform():
    signal = DifficultySignal(context_ids=[0, 1, 2], ema_alpha=1.0, patience=2)
    for _ in range(5):
        events = signal.update(raw_signal={0: 1.0, 1: 1.0, 2: 1.0}, returns={0: 10.0, 1: 10.0, 2: 10.0})
    assert events == []
    assert not signal.is_hard.any()


def test_persistently_bad_context_gets_flagged_after_patience():
    signal = DifficultySignal(context_ids=[0, 1, 2], ema_alpha=1.0, hard_std_factor=0.5, patience=2)
    returns = {0: -100.0, 1: 10.0, 2: 10.0}
    raw = {0: 1.0, 1: 1.0, 2: 1.0}

    events1 = signal.update(raw, returns)
    assert events1 == []  # first bad update, patience not reached yet
    assert not signal.is_hard[0]

    events2 = signal.update(raw, returns)
    assert events2 == [(0, "flagged_hard")]
    assert signal.is_hard[0]


def test_flagged_context_recovers_after_patience_good_updates():
    signal = DifficultySignal(context_ids=[0, 1, 2], ema_alpha=1.0, hard_std_factor=0.5, patience=2)
    bad_returns = {0: -100.0, 1: 10.0, 2: 10.0}
    raw = {0: 1.0, 1: 1.0, 2: 1.0}

    signal.update(raw, bad_returns)
    signal.update(raw, bad_returns)
    assert signal.is_hard[0]

    good_returns = {0: 10.0, 1: 10.0, 2: 10.0}
    events1 = signal.update(raw, good_returns)
    assert events1 == []  # still hard, patience not reached
    assert signal.is_hard[0]

    events2 = signal.update(raw, good_returns)
    assert events2 == [(0, "recovered")]
    assert not signal.is_hard[0]


def test_threshold_needs_at_least_three_seen_contexts():
    signal = DifficultySignal(context_ids=[0, 1], ema_alpha=1.0, hard_std_factor=0.1, patience=1)
    events = signal.update(raw_signal={0: 1.0}, returns={0: -1000.0})
    # only one context has a return so far -> threshold stays -inf -> nothing flagged
    assert events == []
    assert not signal.is_hard.any()


def test_ema_smooths_signal_with_alpha_less_than_one():
    signal = DifficultySignal(context_ids=[0], ema_alpha=0.5)
    signal.update(raw_signal={0: 1.0}, returns={0: 1.0})
    signal.update(raw_signal={0: 3.0}, returns={0: 3.0})
    assert np.isclose(signal.ema_signal[0], 2.0)
    assert np.isclose(signal.ema_return[0], 2.0)
