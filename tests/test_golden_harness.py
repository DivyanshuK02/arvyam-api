import json
from pathlib import Path

PROMPTS = [
    "I’m so sorry for your loss",                 # sympathy (grief_support)
    "Farewell and good luck on your next role",   # farewell (parting_respect)
    "I messed up and I’m truly sorry",            # apology (non-romantic)
    "I’m sorry, my love — I want to make it right", # apology (romantic)
    "Happy birthday to you!",                     # celebration/romance lanes
    "Congratulations on your promotion!",         # celebration
    "Thank you for your kindness",                # selflessness/generosity lane
    "You inspire me every day",                   # intellect/encouragement lanes
]

def test_golden_harness_writes_artifacts(client, evidence_dir):
    manifest = []
    for i, p in enumerate(PROMPTS, 1):
        r = client.post("/api/curate", json={"prompt": p})
        assert r.status_code == 200
        data = r.json()
        # extra guard: public fields only
        for it in data["items"]:
            assert "image" in it and "price" in it and "currency" in it
            assert "image_url" not in it and "price_inr" not in it

        out = evidence_dir / f"{i:02d}.json"
        out.write_text(json.dumps({"prompt": p, "response": data}, ensure_ascii=False, indent=2))
        manifest.append({"file": out.name, "prompt": p, "items": len(data["items"])})

    (evidence_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
