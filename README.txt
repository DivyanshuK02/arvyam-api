ARVYAM — API (Beginner-Safe, Zero-Drift Rails) — Phase 3.1 Snapshot

================================================================================
STATUS: Phase 3.1 Complete
================================================================================
Phase 3.1 (User Accounts & Memory) is deployed and fully tested. All new 
features are feature-flag gated; the default mode behaves as Phase 1.x 
(guest-first, no accounts required). Public /api/curate remains unchanged.

Key docs for maintainers:
- PHASE_3_1_CLOSURE_REPORT.txt — Canonical summary, AC verification, evidence
- PHASE_3_1_HANDOFF_ADDENDUM.txt — Implementation details, flag semantics, ops notes
- __PHASE_3_HANDOFF_GUIDE_-_ARVYAM.txt — Full Phase 3 context and specifications
================================================================================

Deterministic curation API that turns a short user prompt into exactly three products (2 MIX + 1 MONO) from the catalog, each mapped to one of 8 emotional anchors with an explicit palette[].
Stabilized rails (Phase 1.4a + 1.4)
- Single public schema (one return path)
- Family boundaries ON
- Apology relationship_context routing
- Sympathy/Farewell palette sobriety guard
- Enriched rulebook (synonyms, misspellings, combos, India-aware variants)
- Golden harness + contract tests in CI
- Offline LLM feeder (docs + tools) — no runtime LLM

What's new in Phase 1.6A (PR-1 → PR-8)
- PR-1: Hotfix for NameError in unclear-prompt path
- PR-2R: Rules-first detector (Edges → Exact → Combos → Disambig → Keywords)
- PR-3R: JSON-driven edge filters restored (edge_registers.json) with gold-neutral nuance for grief/farewell
- PR-4: Deterministic rotation (prompt-hash seed + tier salts) with recent-ID suppression
What's new in Phase 1.6B (Session-Based Rotation - Technical Debt Resolution)
- Session management: In-memory store with TTL (30 min) and LRU eviction (1000 max)
- Session cookie: arvy_sid (functional, httpOnly, no PII)
- Recent-IDs suppression: Passes recent_ids to engine for rotation via pool filtering
- Deterministic rotation preserved: Same prompt → same first triad across users
- Health endpoint: Added version field (ARVY_VERSION env var, default: v1)
- Tests: test_session_rotation.py validates rotation, determinism, and expiry reset
- Bug Fixed: Repeated curates with same prompt now return different SKUs within session
Constitutional: Phase 1.6 determinism + recent-ID suppression now fully implemented
- PR-5: Pool sizes surfaced for observability
- PR-6: Catalog & rulebook touch-ups (e.g., Affection MONOs → roses), docs/tests aligned
- PR-7: Observability: prompt_hash, pool_sizes, resolved_anchor, edge_type, fallback_reason, final_ids, suppressed_recent_count
- PR-8: Tests & docs refresh (Golden Harness v2, boundary fallback tests, sobriety via single source of truth)

What's new in Phase 3.1 (User Accounts & Memory - Foundations)
- Soft accounts: Email/phone collected per order; user_id nullable; no signup wall
- OTP authentication: /auth/request-otp, /auth/verify-otp with 24h signed tokens (PyJWT)
- OTP security: SHA-256 hashing, 10min TTL, max 5 attempts, 60s cooldown, 20/day per-email cap
- Privacy endpoints: /forget-me (OTP-verified cascade delete), /export-data (OTP-verified JSON export)
- Recipient profiles: /profile endpoints for save/edit lightweight preferences (whitelisted keys only)
- Memory context: Read-only context from last 90 days ({ recent_emotions[], recent_skus[], recipient_prefs[] })
- Post-selection rerank: Optional reranking with bounded weights (≤0.3); deterministic tie-break
- Selection invariance: Memory affects ordering/copy only; SKU IDs and 2 MIX + 1 MONO unchanged
- Feature flags: All memory features OFF by default; surgical rollback via 4 flags
- Database: Supabase-py direct (no ORM); separate tables (users, orders, recipient_profiles, otp_codes)
- Constitutional: catalog.json UNCHANGED (Phase 1.6 freeze); guest-first philosophy maintained
- Tests: test_auth.py, test_privacy.py, test_memory.py, test_invariance.py (71+ tests)

Cross-family fallback (rare): items keep their catalog emotions; we do not forcibly stamp them to the resolved anchor. Tests reflect this contract.
Phase snapshot
1.4a — Stabilization: ✅ complete
1.4 — Emotions & Palettes (Enrichment): ✅ complete
1.5 — Packaging Tiers (internal-only): ✅ complete
1.6 — Catalog Schema Freeze (JSON Schema + CI check): ✅ complete
1.6A — Selection Engine Corrections & Observability (PRs 1–8): ✅ complete
1.6B — Session-Based Rotation (Phase 1 Technical Debt Resolution): ✅ complete
1.7 - BACKEND API: ✅ complete
1.8 - Evidence & Observability (Evidence 2.0) frozen : ✅ complete
3.1 - User Accounts & Memory (Foundations): ✅ complete
Evidence source of truth: stored on device at …/Evidence/ — see Phase-1 §1.8 for structure & manifest rules

Quickstart (local)
# Python 3.11+
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r dev-requirements.txt

# Run API
uvicorn app.main:app --reload
# Open docs: http://127.0.0.1:8000/docs

Public API (contract is frozen by tests)
POST /api/curate
{ "prompt": "Happy birthday to you!"
}

Response (always an array of 3 public items):
[
  {
    "id": "sku_123",
    "title": "Signature Lilies",
    "desc": "Elegant white lilies for timeless wishes",
    "image": "https://.../lilies.jpg",
    "price": 1799,
    "currency": "INR",
    "emotion": "Encouragement/Positivity",
    "palette": ["yellow","orange"],
    "mono": false
  },
  { "... second mix ..." },
  { "... mono ..." }
]

Never returns internal fields (e.g., raw image URLs, packaging, luxury_grand, internal costs).

Auth API (Phase 3.1 — feature-flagged, default OFF)
POST /auth/request-otp
{ "email": "user@example.com" }
→ OTP sent (stub logs to console in dev; real provider in prod)

POST /auth/verify-otp
{ "email": "user@example.com", "otp": "123456" }
→ { "token": "eyJ...", "expires_in": 86400 }

Privacy API (Phase 3.1 — feature-flagged, default OFF)
POST /forget-me — OTP-verified cascade delete (DSAR compliance)
POST /export-data — OTP-verified JSON export (DSAR compliance)
GET /profile — List recipient profiles (JWT required)
POST /profile — Create/update recipient profile (JWT required)

Evidence & Observability (PR-7)
Each request writes a single JSON-line evidence record.
Error JSON shape & rate-limit policy: see docs/deploy.md.
Keys (superset across context+meta):
- request_id (UUID4), prompt_hash
- resolved_anchor (one of 8), edge_type (e.g., valentine, apology, sympathy…)
- relationship_context = romantic|familial|friendship|professional|unknown
- fallback_reason = none|in_family|general_in_family|duplicate_tier|cross_family_last_resort
- pool_sizes = { pre_suppress:{classic,signature,luxury}, post_suppress:{...} }
- suppressed_recent_count
- final_ids (the 3 SKU ids returned)

Environment Variables (Phase 1.6B+)
Session Management:
- SESSION_TTL_SECONDS (default: 1800) — Session expiry time in seconds (30 minutes)
- SESSION_MAX (default: 1000) — Maximum sessions in memory (LRU eviction)
- SESSION_RECENT_PER_ANCHOR (default: 9) — Recent SKUs tracked per emotional anchor

API Configuration:
- ARVY_VERSION (default: v1) — Version string exposed in /health endpoint
- PERSONA_NAME (default: ARVY) — Brand persona for logs/UI
- RATE_LIMIT_PER_MIN (default: 10) — Rate limit per IP address
- ALLOWED_ORIGINS (default: https://arvyam.com) — CORS allowed origins
- ENVIRONMENT (default: development) — Deployment environment (development/production)

Cookie Security:
- secure=True when request.url.scheme == "https" OR ENVIRONMENT == "production"
- secure=False in local development (HTTP)

Environment Variables (Phase 3.1)
Database:
- SUPABASE_URL — Supabase project URL (required for Phase 3.1 features)
- SUPABASE_KEY — Supabase service key (required for Phase 3.1 features)

Auth:
- JWT_SECRET — Random 32+ character string for token signing (required when auth enabled)
- JWT_EXPIRY_HOURS (default: 24) — Token validity period in hours

OTP:
- OTP_PROVIDER (default: stub) — OTP provider: stub (dev), resend, or sendgrid
- OTP_API_KEY — Provider API key (ignored when OTP_PROVIDER=stub)

Feature Flags (all OFF by default):
- AUTH_ENDPOINTS_ENABLED (default: off) — Enable /auth/request-otp, /auth/verify-otp
- MEMORY_ENDPOINTS_ENABLED (default: off) — Enable /forget-me, /export-data, /profile
- MEMORY_CONTEXT_ENABLED (default: off) — Enable memory context building
- MEMORY_RERANK_ENABLED (default: off) — Enable post-selection reranking

Zero-drift acceptance floor
- Exactly 3 items; never 2 or 4
- Public fields only
- Family boundaries enforced; apology context routed
- Grief/Farewell palette sobriety: no celebratory tokens (gold is allowed)
- Deterministic rotation per prompt; recent-ID suppression
- Evidence line stable; CI must be green before merge
- Catalog validates against docs/catalog.schema.json in CI before tests
- Selection invariance: memory cannot change SKU IDs or 2 MIX + 1 MONO composition (Phase 3.1)
- Guest-first: checkout works without signup wall (Phase 3.1)

Repo layout
.
├─ app/
│  ├─ main.py                # FastAPI routes; public transform; evidence writer
│  ├─ selection_engine.py    # Pure logic: detector, edges, boundaries, rotation, triad contract
│  ├─ db.py                  # Supabase client, table constants (Phase 3.1)
│  ├─ rules/
│  │  ├─ emotion_keywords.json
│  │  ├─ edge_registers.json
│  │  ├─ sentiment_families.json
│  │  └─ tier_policy.json
│  ├─ accounts/              # Phase 3.1: Auth & Privacy
│  │  ├─ __init__.py
│  │  ├─ otp.py              # OTPManager, stub/real providers, daily cap
│  │  ├─ auth.py             # JWT create/verify, signed tokens
│  │  ├─ models.py           # Pydantic schemas, validators, whitelists
│  │  ├─ routes.py           # Privacy endpoints (/forget-me, /export-data, /profile)
│  │  └─ auth_routes.py      # Auth endpoints (/auth/request-otp, /auth/verify-otp)
│  └─ memory/                # Phase 3.1: Memory Context & Rerank
│     ├─ __init__.py
│     ├─ context.py          # build_context(), MemoryContext TypedDict, 90-day lookback
│     └─ rerank.py           # rerank(), RerankedTriad, bounded weights, invariance check
├─ migrations/               # Phase 3.1: Database migrations
│  └─ 001_phase_3_1.sql      # users, orders, recipient_profiles, otp_codes tables
├─ tests/
│  ├─ test_api_contract.py         # Public shape + 3-card invariant
│  ├─ test_golden_harness.py       # v2: anchors/edges, valentine mono-rose, sobriety, artifacts
│  ├─ test_family_boundaries.py    # CELEBRATION_BLOCK (gold allowed) — exported constant
│  ├─ test_boundary_fallbacks.py   # fallback reasons, pool sizes; 1 MONO invariant; post<=pre
│  ├─ test_apology_context.py
│  ├─ test_apply_enums.py
│  ├─ test_miner_filter.py
│  ├─ test_session_rotation.py     # Session rotation, determinism, expiry reset (P1.6B)
│  ├─ test_packaging_tiers.py
│  ├─ test_auth.py                 # Phase 3.1: OTP, JWT, daily cap, constant-time comparison
│  ├─ test_privacy.py              # Phase 3.1: /forget-me, /export-data, profile CRUD, masking
│  ├─ test_memory.py               # Phase 3.1: Memory context, 90-day lookback, opt-in flag
│  └─ test_invariance.py           # Phase 3.1: Selection invariance (AC3 proof), bounded weights
├─ tools/
│  ├─ mine_unknowns.py         # Feeder: mine unknown phrases from logs (offline)
│  ├─ make_review_sheet.py     # Feeder: candidates/proposals → review.csv
│  ├─ apply_tokens.py          # Feeder: add-only to emotion_keywords.json (+ .bak)
│  ├─ validate_rulebook.py     # Schema/enum validator for rulebook
│  ├─ validate_catalog.py      # Validates app/catalog.json against docs/catalog.schema.json
│  └─ rotate_backups.py        # Keep newest N .bak files (local hygiene)
├─ docs/
│  ├─ llm_feeder.md            # Offline feeder guide (no runtime LLM)
│  ├─ phase_status.json        # Live phase ticks
│  └─ catalog.schema.json      # Frozen catalog JSON Schema (P1.6)
├─ evidence/                   # Golden harness artifacts (gitignored)
├─ requirements.txt
└─ dev-requirements.txt

Tests & CI
Local:
pytest -q

CI runs (order):
- Rulebook schema check (tools/validate_rulebook.py)
- Feeder smoke dry-run (tools/apply_tokens.py on tests/data/review_smoke.csv)
- Catalog JSON Schema check (tools/validate_catalog.py against docs/catalog.schema.json)
- Pytest suite:
  • Contract + public-shape (3-card rule)  
  • Golden Harness v2 (artifacts under evidence/)  
  • Boundary fallbacks (reasons, pool size monotonicity, 1 MONO invariant)  
  • Palette sobriety using CELEBRATION_BLOCK  
  • Enums & feeder smokes (no runtime LLM)
  • Phase 3.1: Auth (OTP, JWT, daily cap)
  • Phase 3.1: Privacy (cascade delete, export, profile CRUD)
  • Phase 3.1: Memory (context building, opt-in, 90-day lookback)
  • Phase 3.1: Selection invariance (SKU set unchanged, bounded weights, determinism)
- Artifacts uploaded: evidence-bundle, review_smoke_csv

Removed legacy test_palette_allowlist.py and redundant sympathy guard file; sobriety is enforced via test_family_boundaries.py and used by Golden Harness v2.

Behavior notes (for reviewers)
- Detector is rules-first; keywords are a last resort.
- Edges (valentine/apology/sympathy/…) read from edge_registers.json (gold-neutral grief/farewell).
- Rotation is deterministic per prompt (CRC32 prompt hash + tier salts).
- Cross-family fallback: when anchor pools are exhausted, we prefer anchor-coherent items; if still insufficient, we may loosen—items keep catalog emotions (we don't restamp to the resolved anchor). Tests assert the contract.
- Memory rerank (Phase 3.1): Post-selection only; bounded weights (≤0.3); deterministic tie-break (weight DESC, original_index ASC, sku_id ASC); SKU IDs unchanged.

Offline feeder (docs + tools)
Guide: docs/llm_feeder.md (beginner-proof; add-only; no runtime LLM).
Reproduce CI's smoke locally (sanity-check headers/enums):
python tools/apply_tokens.py --review tests/data/review_smoke.csv --rules app/rules/emotion_keywords.json --dry-run

Backups: tools/apply_tokens.py creates timestamped .bak next to the rulebook.
Local hygiene (optional):
python tools/rotate_backups.py --keep 10

Troubleshooting
- "Cannot import app in tests": run from repo root with venv active.
- Golden Harness mismatch on valentine MONO: ensure the MONO SKU contains rose (catalog).
- Sobriety failures in grief/farewell: verify palettes vs CELEBRATION_BLOCK.
- Unexpected fallback: check fallback_reason, anchor_filter_loosened, and pool_sizes in evidence.
- Import error in tests (cannot find app): run from repo root with an active venv.
- CI fails on feeder smoke: verify tests/data/review_smoke.csv headers and anchor enums.
- Engine behavior: confirm edge overrides (e.g., romantic apology → Affection/Support) are honored in selection_engine.py per edge_registers.json.
- Phase 3.1 test failures: Verify imports match actual exports (e.g., OTP_DAILY_CAP_PER_EMAIL not DAILY_OTP_CAP); mock paths match import locations (e.g., @patch('app.db.get_supabase_client')); test data uses whitelisted values.

Contribution notes
- Keep changes additive in app/rules/emotion_keywords.json (Phase 1.4 policy).
- Do not weaken the rails (public schema, 3 cards, boundaries, apology context).
- If you touch rules or tests, run the feeder smoke command above before pushing.
- Phase 3.1: catalog.json remains FROZEN; user/memory data in separate tables only.
- Phase 3.1: Selection invariance must hold (memory affects ordering/copy, not SKU IDs).

Working Rules:
Active doc rule: When multiple versions exist in this project thread, the last uploaded file with the same base name is the authoritative version unless a different version is explicitly referenced in the task.
### One-Pass Review Policy (Skills & Specs)
- All reviewer feedback must be consolidated into **a single, comprehensive package per revision** (MUST-FIX + Polishes + Nits).
- Subsequent feedback is allowed **only** for:
  1) P0 correctness/security/privacy issues, or
  2) Conflicts with constitutional rails (1.6, 1.7, 2.1, 2.3, 2.5, 3.1).
- Otherwise, the artifact ships and iteration moves to the next version tag.

Release tags
v1.6 — Catalog Schema Freeze (p1.6-schema-freeze)
v1.6A — Selection Engine Corrections & Observability (PRs 1–8)
v1.6B — Session-Based Rotation Fix (p1.6b-session-rotation)
v3.1 — User Accounts & Memory Foundations (p3.1-user-accounts-memory)
