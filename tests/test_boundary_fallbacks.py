# tests/test_boundary_fallbacks.py
import pytest
from app.selection_engine import selection_engine
from typing import Any, Dict, List

def _ctx():
    return {}

def test_family_boundary_fallbacks_expose_reason():
    # generic prompt to avoid edges; ensures engine builds family pool
    items, ctx, _ = selection_engine(prompt="a nice bouquet", context=_ctx())
    assert isinstance(items, list) and len(items) == 3
    # reason should always be present
    assert "fallback_reason" in ctx
    assert ctx["fallback_reason"] in {"none","in_family","general_in_family","duplicate_tier","cross_family_last_resort"}
    # PR-8A.2 invariant: exactly one MONO in the final triad
    mono_count = sum(1 for it in items if bool(it.get("mono")))
    assert mono_count == 1, f"Expected exactly one MONO, got {mono_count}"

def test_pool_size_counts_present():
    _, ctx, _ = selection_engine(prompt="birthday party", context=_ctx())
    ps = ctx.get("pool_size")
    assert isinstance(ps, dict)
    assert "pre_suppress" in ps and "post_suppress" in ps
    for k in ("classic","signature","luxury"):
        assert k in ps["pre_suppress"]
        assert k in ps["post_suppress"]
        # PR-8A.2: post-suppress cannot exceed pre-suppress
        assert ps["post_suppress"][k] <= ps["pre_suppress"][k]

def test_cross_family_fallback_emotion_and_loosen_flag():
    """
    When we fall back across families, items still carry an emotion (catalog or stamped),
    and the engine may expose that the anchor filter was loosened.
    """
    prompt = "starlight river pebble whisper"
    items, ctx, meta = selection_engine(prompt=prompt, context=_ctx())
    assert isinstance(items, list) and len(items) == 3
    # Contract: exactly one MONO
    assert sum(1 for it in items if it.get("mono")) == 1
    # Tolerant check for cross-family fallback (by design we preserve catalog emotion)
    if ctx.get("fallback_reason") == "cross_family_last_resort":
        assert all(bool(it.get("emotion")) for it in items), "Each card must have an emotion"
        # Loosen flag is present only when we had to widen the pool
        assert ctx.get("anchor_filter_loosened") in (None, True)
