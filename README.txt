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
│     ├─ pricing_policy.json  # optional
│     └─ weights_default.json # optional
├─ tests/
│  ├─ conftest.py
│  ├─ test_api_contract.py      # public-schema contract (typed POST)
│  ├─ test_golden_harness.py    # multi-prompt evidence writer
│  ├─ test_family_boundaries.py  # drift guard (sympathy/farewell)
│  └─ test_apology_context.py    # romantic vs non-romantic routing
├─ evidence/                  # golden harness outputs (GITIGNORED)
├─ requirements.txt           # runtime deps (FastAPI, Uvicorn, etc.)
├─ dev-requirements.txt       # test-only deps (pytest, httpx)
├─ .github/workflows/ci.yml   # CI: install + run pytest (+ upload artifacts)
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
- Returns: a list of exactly 3 public items:
  [
    { "id","title","desc","image","price","currency" },
    { ... },
    { ... }
  ]
Notes:
- Public payload uses image/price/currency (never image_url/price_inr).
- Always exactly 3 items.

Error Shape (structured)
{
  "error": { "code": "INPUT_TOO_LONG", "message": "Max 240 chars" },
  "persona": "ARVY",
  "request_id": "<uuid>"
}

Golden Harness & Evidence
- The golden harness test writes one JSON file per prompt under:
  evidence/p1_4a_harness_<RUN>/
- The evidence/ directory is **root-level and gitignored** to keep the repo clean.
- CI uploads the evidence folder as a run artifact for review.

Logging / Telemetry
- The engine emits a single-line JSON “SELECTION_EVIDENCE” per request with:
  request_id, resolved_anchor, relationship_context, pool_size {pre_suppress,post_suppress}, and fallback_reason ∈ {"in_family","general_in_family","duplicate_tier","cross_family_last_resort"}.
- Use these to spot catalog scarcity and verify boundary behavior.

Tests (PR-4 guards)
1) Contract test (public schema)
   - pytest -q -k contract
2) Golden harness (evidence pack)
   - pytest -q -k golden
3) Family boundary tests (no celebration palettes for sympathy/farewell)
   - pytest -q -k family_boundaries
4) Apology routing tests (romantic vs non-romantic)
   - pytest -q -k apology_context

CI (GitHub Actions)
- Runs on every push/PR
- Installs runtime + dev deps, executes pytest
- Uploads evidence/ as a build artifact (see .github/workflows/ci.yml)

Troubleshooting
- 404 on “HEAD / HTTP/1.1” in Render logs is harmless (health check).
- Ensure PERSONA_NAME=ARVY is present; CORS uses ALLOWED_ORIGINS.
- If CI fails, check: dependency pins, schema drift (contract test), or evidence paths.

Ownership & Source of Truth
- main.py defines routes; selection_engine.py is logic-only.
- Policy lives under app/rules/*. Keep engine + policy **in sync**.
