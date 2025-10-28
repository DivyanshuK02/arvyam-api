# tests/test_family_boundaries.py
import json
from pathlib import Path

# --- Helper function (Restored to remove 'utils' dependency) ---

def _load_catalog():
    # Assumes pytest runs from the project root, finding app/catalog.json
    try:
        with Path("app/catalog.json").open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback for different CWD (e.g., running from inside /tests)
        with Path("../app/catalog.json").open("r", encoding="utf-8") as f:
            return json.load(f)

# --- P1.6 Test Logic ---

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
