# tests/test_family_boundaries.py
import json
from pathlib import Path

GRIEF_FAMILIES = {"grief_support", "parting_respect"}
CELEBRATION_BLOCK = {"deep-red", "crimson", "gold", "neon", "bright", "hot-pink"}

def _load_catalog():
    with Path("app/catalog.json").open("r", encoding="utf-8") as f:
        return json.load(f)

def test_grief_and_farewell_palettes_are_sober():
    for row in _load_catalog():
        if row.get("sentiment_family") in GRIEF_FAMILIES:
            chips = {
                (p or "").strip().lower()
                for p in (row.get("palette") or [])
                if isinstance(p, str)
            }
            bad = CELEBRATION_BLOCK & chips
            assert not bad, f"{row.get('id')} uses celebratory colors in grief/farewell: {sorted(bad)}"
