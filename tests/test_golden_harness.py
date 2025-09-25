# tests/test_golden_harness.py
import json, uuid
from pathlib import Path
from datetime import datetime

from app.selection_engine import selection_engine  # for context evidence
# client + evidence_dir fixtures come from tests/conftest.py

# 8 anchors + 2 edges (apology romantic / sympathy)
GOLDEN_CASES = [
    ("affection_support",        "romantic bouquet for date night"),
    ("loyalty_dependability",    "farewell note for teammate transfer"),
    ("encouragement_positivity", "good luck for board exams"),
    ("strength_resilience",      "so sorry for your loss, deepest condolences"),
    ("intellect_wisdom",         "thank you teacher for guidance"),
    ("adventurous_creativity",   "housewarming gift for new home"),
    ("selflessness_generosity",  "thanks a ton for helping"),
    ("fun_humor",                "fun birthday party vibes"),
    ("edge_apology_romantic",    "I am so sorry my love, I messed up completely"),
    ("edge_sympathy",            "I’m so sorry for your loss"),
]

REQUIRED_PUBLIC_KEYS = (
    "id", "title", "desc", "image", "price", "currency",
    "emotion", "tier", "mono", "palette"
)
FORBIDDEN_KEYS = {
    "packaging","luxury_grand","image_url","price_inr","flowers","weight","tags"
}

# Editorial mapping (internal packaging stays hidden; UI label is derived from tier)
DISPLAY_LABEL = {"Classic": "Box", "Signature": "Vase", "Luxury": "Premium Box"}


def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def test_golden_harness_writes_artifacts(client, evidence_dir):
    """
    Hits the public API with canonical prompts (8 anchors + 2 edges),
    asserts the rails, and saves evidence artifacts under evidence/.
    """
    manifest = []

    for label, prompt in GOLDEN_CASES:
        # 1) Call API (public contract)
        r = client.post("/api/curate", json={"prompt": prompt})
        assert r.status_code == 200

        items = r.json()
        assert isinstance(items, list), "Public response must be a list"
        assert len(items) == 3, "Exactly 3 items (2 MIX + 1 MONO)"

        # Guard: public fields only, no raw catalog or private fields
        mono_count = 0
        seen_tiers = set()
        ui_labels = []

        for it in items:
            # must-have keys
            for k in REQUIRED_PUBLIC_KEYS:
                assert k in it, f"Missing public field '{k}'"
            # no forbidden or underscored keys
            for fk in FORBIDDEN_KEYS:
                assert fk not in it, f"forbidden key leaked: {fk}"
            assert not any(str(k).startswith("_") for k in it.keys()), "underscored key leaked"
            # palette shape + currency
            assert isinstance(it["palette"], list) and len(it["palette"]) > 0, "palette[] must be non-empty"
            assert it["currency"] == "INR"

            # editorial label (derived from tier)
            tier = it["tier"]
            assert tier in DISPLAY_LABEL, f"unexpected tier value: {tier}"
            ui_labels.append(f"{tier} {DISPLAY_LABEL[tier]}")
            seen_tiers.add(tier)

            if it.get("mono") is True:
                mono_count += 1
        assert mono_count == 1, "There must be exactly 1 MONO card"

        # If any Luxury appears, its label must be 'Premium Box' (with a space)
        if "Luxury" in seen_tiers:
            assert "Luxury Premium Box" in ui_labels

        # 2) Get context (for evidence only; runtime contract is API above)
        _, ctx, _ = selection_engine(prompt=prompt, context={})
        ctx_light = {
            "resolved_anchor": ctx.get("resolved_anchor"),
            "relationship_context": ctx.get("relationship_context"),
            "edge_type": ctx.get("edge_type"),
            "sentiment_family": ctx.get("sentiment_family"),
            "pool_size": {
                "pre_suppress": ctx.get("pool_size", {}).get("pre_suppress"),
                "post_suppress": ctx.get("pool_size", {}).get("post_suppress"),
            },
            "fallback_reason": ctx.get("fallback_reason"),
        }

        # 3) Save artifacts per case
        case_id = f"{label}_{uuid.uuid4().hex[:6]}"
        ts = datetime.utcnow().isoformat() + "Z"

        _save_json(evidence_dir / f"{case_id}.response.json", {
            "prompt": prompt,
            "timestamp_utc": ts,
            "items": items,
            "ui_labels": ui_labels,  # editorial labels for auditors
        })
        _save_json(evidence_dir / f"{case_id}.context.json", {
            "prompt": prompt,
            "timestamp_utc": ts,
            "context": ctx_light,
        })

        manifest.append({
            "case": label,
            "prompt": prompt,
            "timestamp_utc": ts,
            "files": {
                "response": str((evidence_dir / f"{case_id}.response.json").as_posix()),
                "context": str((evidence_dir / f"{case_id}.context.json").as_posix()),
            },
            "counts": {
                "items": len(items),
                "mono_cards": mono_count,
            },
        })

    # 4) Write manifest last
    _save_json(evidence_dir / "manifest.json", manifest)
