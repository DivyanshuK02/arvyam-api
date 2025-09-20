ARVYAM — Deployment & Rails (Beginner-Safe)

Persona (Canonical)
- Persona name: ARVY (single source of truth)
- Environment variable: PERSONA_NAME=ARVY
- UI caps: 240 chars (short input), 500 chars (long input in future)
- Error JSON must include: "persona":"ARVY"

Project Layout (execution-ready, zero-drift)
.
├─ app/
│  ├─ main.py                 # single FastAPI app & routes
│  ├─ selection_engine.py     # pure logic module (no routes)
│  ├─ catalog.json            # merchandise catalog
│  └─ rules/
│     ├─ edge_registers.json
│     ├─ emotion_keywords.json
│     ├─ sentiment_families.json
│     ├─ tier_policy.json
│     ├─ pricing_policy.json (optional)
│     └─ weights_default.json (optional)
├─ tests/
│  ├─ conftest.py
│  ├─ test_api_contract.py    # public-schema contract (typed POST)
│  └─ test_golden_harness.py  # multi-prompt evidence writer
├─ evidence/                  # golden harness outputs (gitignored)
├─ requirements.txt           # runtime deps (FastAPI, Uvicorn, etc.)
├─ dev-requirements.txt       # test-only deps (pytest, etc.)
├─ .github/workflows/ci.yml   # CI: install + run pytest
├─ .gitignore                 # ignores evidence/ & caches
└─ README.txt                 # this file

How to Run (Render / No CLI)
1) Render → Settings → Environment — add:
   - PERSONA_NAME=ARVY
   - ALLOWED_ORIGINS=<your frontend origin>
   - PUBLIC_ASSET_BASE=/assets
2) Click “Save changes” → “Deploy latest commit”.

Build
- pip install -r requirements.txt

Start (Render uses this)
- uvicorn app.main:app --host 0.0.0.0 --port $PORT

Local Dev (optional)
- python -m venv .venv && source .venv/bin/activate
- pip install -r requirements.txt -r dev-requirements.txt
- uvicorn app.main:app --reload

API Contracts (zero-drift)
POST /api/curate
- Body: { "prompt": "<text ≤ 240 chars>" }
- Returns: { "items": [ { id,title,desc,image,price,currency } x3 ], "edge_case": <bool> }

Notes:
- Public payload uses image/price/currency (never image_url/price_inr).
- Always exactly 3 items.

Error Shape (structured)
{
  "error": { "code": "INPUT_TOO_LONG", "message": "Max 240 chars" },
  "persona": "ARVY",
  "request_id": "<uuid>"
}

Tests (PR-4)
1) Contract test (public schema)
   - pytest -q -k contract
2) Golden harness (evidence pack)
   - pytest -q -k golden
   - JSON files written under: evidence/p1_4a_harness_<RUN>/

CI (GitHub Actions)
- Runs on every push/PR
- Installs runtime + dev deps, executes pytest
- Optional: upload golden harness artifacts (see ci.yml snippet below)

Logging / Evidence
- Selection evidence and telemetry emitted by the engine (stdout)
- Golden harness writes readable JSON artifacts per prompt to evidence/

Troubleshooting
- 404 on “HEAD / HTTP/1.1” in Render logs is harmless (health check).
- Ensure PERSONA_NAME=ARVY is present; CORS uses ALLOWED_ORIGINS.
- If CI fails, check: missing deps, changed schema, or test prompt set.

Ownership
- main.py defines routes; selection_engine.py is logic-only.
- Policy lives in app/rules/* and is synced with the engine.

