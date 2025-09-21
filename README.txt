ARVYAM — Deployment, Contracts & Zero-Drift Runbook
===================================================

Persona (single source of truth)
- Persona name: ARVY
- ENV: PERSONA_NAME=ARVY
- Public errors must include: "persona":"ARVY"

Zero-Drift Principles (enforced)
- One FastAPI app (app/main.py) – routes live here only
- selection_engine.py is logic-only (no routes)
- Public API always returns the same v1 contract (see below)
- Tests + CI guard the contract and the evidence bundle

Repository Layout (execution-ready)
.
├─ app/
│  ├─ main.py                  # single FastAPI app & routes
│  ├─ selection_engine.py      # pure logic module (no routes)
│  ├─ catalog.json             # merchandise catalog (public fields: image/price/currency)
│  └─ rules/
│     ├─ edge_registers.json
│     ├─ emotion_keywords.json
│     ├─ sentiment_families.json
│     ├─ tier_policy.json
│     ├─ weights_default.json
│     └─ pricing_policy.json   # optional
├─ tests/
│  ├─ conftest.py
│  ├─ test_api_contract.py        # public-schema contract
│  ├─ test_golden_harness.py      # evidence pack generator
│  ├─ test_family_boundaries.py   # no grief→celebration drift
│  └─ test_apology_context.py     # romantic vs non-romantic routing
├─ evidence/                   # golden outputs (gitignored; folder kept via .keep)
├─ requirements.txt            # runtime deps
├─ dev-requirements.txt        # test/harness deps (pytest, httpx, etc.)
├─ .github/workflows/ci.yml    # CI: install deps + run pytest (+ upload evidence)
├─ .gitignore                  # ignores evidence/, venv, caches, .env, logs
└─ README.txt                  # this file

How to Run (Local)
1) python -m venv .venv && source .venv/bin/activate
2) pip install -r requirements.txt -r dev-requirements.txt
3) export PERSONA_NAME=ARVY
4) uvicorn app.main:app --reload

How to Run (Render / container)
- Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
- ENV:
  - PERSONA_NAME=ARVY
  - ALLOWED_ORIGINS=<your frontend origin>
  - PUBLIC_ASSET_BASE=/assets

Public API Contract (v1)
POST /api/curate
Request:
  { "prompt": "<text ≤ 240 chars>" }
Response:
  [
    {
      "id": "...",
      "title": "...",
      "desc": "...",
      "image": "<public URL>",
      "price": <number>,
      "currency": "INR",
      "palette": ["..."],        // array of tokens
      "mono": <true|false>,      // exactly 1 item must be mono
      "tier": "Classic|Signature|Luxury",
      "emotion": "<anchor>"      // from catalog or fallback to resolved anchor
    },
    { ... }, { ... }             // ALWAYS exactly 3 items
  ]
Notes:
- Never expose internal fields (image_url, price_inr). They’re mapped to public fields by the transformer.
- Edge rails and palettes are enforced by the engine.

Error Shape (structured)
{
  "error": { "code": "INPUT_TOO_LONG" | "BAD_REQUEST" | "INTERNAL", "message": "..." },
  "persona": "ARVY",
  "request_id": "<uuid>"
}

Selection Evidence (one JSON line per request)
- Event name: SELECTION_EVIDENCE
- Required keys:
  {
    "request_id": "<uuid4>",
    "resolved_anchor": "<anchor>",
    "relationship_context": "romantic|familial|friendship|professional|unknown",
    "pool_size": {
      "pre_suppress": {"classic": n, "signature": n, "luxury": n},
      "post_suppress": {"classic": n, "signature": n, "luxury": n}
    },
    "fallback_reason": "in_family|general_in_family|duplicate_tier|cross_family_last_resort"
  }

Golden Harness (evidence pack)
- tests/test_golden_harness.py hits 6–10 canonical prompts (8 anchors + ≥2 edges)
- Outputs JSON files into evidence/p1_4a_harness_<RUN>/
- CI uploads the evidence/ directory as an artifact

Tests (must stay green)
- Contract: pytest -q -k api_contract
- Golden:   pytest -q -k golden
- Family:   pytest -q -k family_boundaries
- Apology:  pytest -q -k apology_context

CI (GitHub Actions)
- Installs runtime + dev deps
- Runs pytest
- Uploads evidence/ as artifact
- Blocks merges on test failures (prevents schema drift)

Troubleshooting
- 404 on HEAD / in Render logs: harmless health check
- 500s: check that _transform_for_api is used on ALL returned items
- Schema failures: contract test will point to the missing field
- Evidence missing: ensure evidence/ exists (repo keeps empty folder via .keep), and CI has artifact upload step

Ownership & Source of Truth
- Routes: app/main.py
- Logic: app/selection_engine.py
- Policy/rules: app/rules/*.json
- Keep engine + policy JSONs in sync; any policy change must be paired with a test update

Versioning Note
- v1 is frozen. New fields must be additive/optional.
- If you need breaking changes, introduce /v2 and keep v1 active until sunset.

