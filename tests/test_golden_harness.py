import os, json
from fastapi.testclient import TestClient
from app.main import app

# A subset of prompts from the main golden harness for quick checks
PROMPTS = [
    "romantic anniversary under 2000",
    "i’m so sorry for your loss",
    "i deeply apologize",
    "bright congratulations",
]

def test_golden_harness_writes_artifacts(client, evidence_dir):
    manifest = []
    for i, p in enumerate(PROMPTS, 1):
        r = client.post("/api/curate", json={"prompt": p})
        assert r.status_code == 200
        
        # FIX: The response (data) is the list itself.
        items = r.json()

        # extra guard: public fields only
        for it in items:
            assert "image_url" not in it
            assert "price_inr" not in it
            assert "image" in it
            assert "price" in it
