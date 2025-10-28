# tests/test_family_boundaries.py
from utils import _load_catalog  # or your fixture

CELEBRATION_BLOCK = {"crimson", "deep-red", "hot-pink", "neon", "fuchsia"}  # gold allowed
GRIEF_FAMILIES = {"grief", "farewell"}

def test_grief_and_farewell_palettes_are_sober():
    for row in _load_catalog():
        if row.get("sentiment_family") in GRIEF_FAMILIES:
            pal = {(p or "").strip().lower() for p in (row.get("palette") or []) if isinstance(p, str)}
            assert pal.isdisjoint(CELEBRATION_BLOCK), (
                f"{row.get('id')} uses celebratory pops in grief/farewell: "
                f"{sorted(pal & CELEBRATION_BLOCK)}"
            )

