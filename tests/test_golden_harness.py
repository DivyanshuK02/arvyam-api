import json
from pathlib import Path

# The original, richer set of prompts to cover more scenarios
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

        # FIX: The response from r.json() is the list of items directly.
        items_list = r.json()
        
        # Guard: public fields only
        for it in items_list:
            assert "image" in it and "price" in it and "currency" in it
            assert "image_url" not in it and "price_inr" not in it

        # Write the full response for this prompt to its own artifact file
        out = evidence_dir / f"{i:02d}.json"
        out.write_text(json.dumps({"prompt": p, "response": items_list}, ensure_ascii=False, indent=2))
        
        # Add an entry to the manifest for this run
        manifest.append({"file": out.name, "prompt": p, "items": len(items_list)})

    # Write the summary manifest file
    (evidence_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
