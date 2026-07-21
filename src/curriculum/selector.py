"""Bridges a CurriculumController to CARL's context-selection hook."""
from __future__ import annotations

import numpy as np
from carl.context.selection import CustomSelector


def build_curriculum_selector(contexts: dict[int, dict], controller) -> CustomSelector:
    """Every reset, sample a context_id according to the controller's current weights."""
    context_ids = list(contexts.keys())

    def _selector_fn(selector: CustomSelector):
        weights = controller.get_weights()
        cid = int(np.random.choice(context_ids, p=weights))
        return selector.contexts[cid], cid

    return CustomSelector(contexts, _selector_fn)


def build_fixed_context_selector(contexts: dict[int, dict], context_id: int) -> CustomSelector:
    """Always returns the same context_id (used for deterministic held-out eval)."""

    def _selector_fn(selector: CustomSelector):
        return selector.contexts[context_id], context_id

    return CustomSelector(contexts, _selector_fn)
