# tests/test_boundary_fallbacks.py
import pytest
from app.selection_engine import selection_engine

def _ctx():
    return {}

def test_family_boundary_fallbacks_expose_reason():
    # generic prompt to avoid edges; ensures engine builds family pool
    items, ctx, _ = selection_engine(prompt="a nice bouquet", context=_ctx())
    assert isinstance(items, list) and len(items) == 3
    # reason should always be present
    assert "fallback_reason" in ctx
    assert ctx["fallback_reason"] in {"none","in_family","general_in_family","duplicate_tier","cross_family_last_resort"}

def test_pool_size_counts_present():
    _, ctx, _ = selection_engine(prompt="birthday party", context=_ctx())
    ps = ctx.get("pool_size")
    assert isinstance(ps, dict)
    assert "pre_suppress" in ps and "post_suppress" in ps
    for k in ("classic","signature","luxury"):
        assert k in ps["pre_suppress"]
        assert k in ps["post_suppress"]
