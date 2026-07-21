import numpy as np
import pytest

from src.curriculum.controller import CurriculumController


def test_ignore_policy_never_changes_weights():
    controller = CurriculumController(context_ids=[0, 1, 2], handling_policy="ignore")
    before = controller.get_weights().copy()
    controller.update_from_signal(raw_signal={0: 100.0, 1: 0.0, 2: 0.0}, returns={0: 1.0, 1: 1.0, 2: 1.0})
    after = controller.get_weights()
    assert np.array_equal(before, after)


def test_replay_priority_updates_weights_and_history():
    controller = CurriculumController(context_ids=[0, 1, 2], handling_policy="replay_priority")
    controller.update_from_signal(raw_signal={0: 10.0, 1: 0.1, 2: 0.1}, returns={0: 1.0, 1: 1.0, 2: 1.0})
    weights = controller.get_weights()
    assert np.isclose(weights.sum(), 1.0)
    assert weights[0] > weights[1]
    assert len(controller.weight_history) == 1


def test_hard_exclude_downweights_flagged_context():
    controller = CurriculumController(
        context_ids=[0, 1, 2], handling_policy="hard_exclude", ema_alpha=1.0, hard_std_factor=0.5, patience=1
    )
    for _ in range(2):
        events = controller.update_from_signal(
            raw_signal={0: 1.0, 1: 1.0, 2: 1.0}, returns={0: -100.0, 1: 10.0, 2: 10.0}
        )
    assert ("flagged_hard" in [e for _cid, e in events]) or controller.signal.is_hard[0]
    weights = controller.get_weights()
    assert weights[0] < weights[1]


def test_downweight_revisit_and_events_returned():
    controller = CurriculumController(
        context_ids=[0, 1, 2],
        handling_policy="downweight_revisit",
        ema_alpha=1.0,
        hard_std_factor=0.5,
        patience=1,
        cooldown=2,
    )
    events = controller.update_from_signal(
        raw_signal={0: 1.0, 1: 1.0, 2: 1.0}, returns={0: -100.0, 1: 10.0, 2: 10.0}
    )
    assert (0, "flagged_hard") in events


def test_unknown_policy_raises():
    with pytest.raises(ValueError):
        CurriculumController(context_ids=[0, 1], handling_policy="not-a-policy")


def test_weights_always_normalized():
    controller = CurriculumController(context_ids=[0, 1, 2], handling_policy="replay_priority")
    for i in range(10):
        controller.update_from_signal(raw_signal={0: float(i), 1: 1.0, 2: 2.0}, returns={0: 1.0, 1: 1.0, 2: 1.0})
        assert np.isclose(controller.get_weights().sum(), 1.0)
