from app.selection_engine import selection_engine
import pytest

CELEBRATION_PALETTE = {"deep-red", "crimson", "gold", "bright", "hot-pink", "neon"}

PROMPTS_TO_GUARD = [
    "I’m so sorry for your loss",
    "condolences on your loss",
    "Farewell and good luck in your next role",
    "farewell to a colleague",
]

@pytest.mark.parametrize("prompt", PROMPTS_TO_GUARD)
def test_guarded_prompts_never_use_celebration_palettes(prompt):
    """Verifies that sympathy/farewell prompts do not return items with celebration palettes."""
    items, _, _ = selection_engine(prompt=prompt, context={})
    assert len(items) == 3, "Expected a triad of 3 items"
    
    for item in items:
        palette = set(item.get("palette", []))
        forbidden_found = palette & CELEBRATION_PALETTE
        assert not forbidden_found, \
            f"Item {item.get('id')} for prompt '{prompt}' has forbidden palette token(s). "\
            f"Found: {forbidden_found}"
