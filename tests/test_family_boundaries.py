# tests/test_family_boundaries.py
import json
from pathlib import Path

GRIEF = {"grief_support"}
FAREWELL = {"parting_respect"}

# Grief: strictest block (gold is disallowed)
GRIEF_BLOCK = {"deep-red", "crimson", "gold", "neon", "bright", "hot-pink"}

# Farewell: still sober, but gold is permitted
FAREWELL_BLOCK = {"deep-red", "crimson", "neon", "bright", "hot-pink"}

def _load_catalog():
    with Path("app/catalog.json").open("r", encoding="utf-8") as f:
        return json.load(f)

def _chips(row):
    return {
        (p or "").strip().lower()
        for p in (row.get("palette") or [])
        if isinstance(p, str)
    }

def test_grief_palettes_are_sober():
    for row in _load_catalog():
        if row.get("sentiment_family") in GRIEF:
            bad = _chips(row) & GRIEF_BLOCK
            assert not bad, f"{row.get('id')} grief/support palette contains disallowed: {sorted(bad)}"

def test_farewell_palettes_are_sober():
    for row in _load_catalog():
        if row.get("sentiment_family") in FAREWELL:
            bad = _chips(row) & FAREWELL_BLOCK
            assert not bad, f"{row.get('id')} farewell palette contains disallowed: {sorted(bad)}"

