# main.py
import os, json, logging, uuid, time, csv
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

# Rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded

from .selection_engine import selection_engine, normalize, detect_emotion, _transform_for_api

# =========================
# Environment & Constants
# =========================
ALLOWED = os.getenv("ALLOWED_ORIGINS", "https://arvyam.com")
ASSET_BASE = os.getenv("PUBLIC_ASSET_BASE", "https://arvyam.com").rstrip("/")
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "10"))
PERSONA = os.getenv("PERSONA_NAME", "ARVY")  # for logs/UI
ERROR_PERSONA = "ARVY"                       # hard-coded in API errors
ICONIC = {"rose","lily","orchid"}            # for analytics guard
ANALYTICS_ENABLED = os.getenv("ANALYTICS_ENABLED", "0") == "1"
GOLDEN_ARTIFACTS_DIR = os.getenv("GOLDEN_ARTIFACTS_DIR", "/tmp/arvy_golden")


# =========================
# Logging
# =========================
logger = logging.getLogger("arvyam")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# =========================
# App Setup
# =========================
limiter = Limiter(key_func=get_remote_address, default_limits=[f"{RATE_LIMIT_PER_MIN}/minute"])
app = FastAPI(title="Arvyam API")
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED.split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Data Loaders
# =========================
HERE = os.path.dirname(__file__)

def load_json(relpath: str, default=None):
    try:
        with open(os.path.join(HERE, relpath), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

CATALOG: List[Dict[str, Any]] = load_json("catalog.json", default=[])
CAT_BY_ID: Dict[str, Dict[str, Any]] = {it["id"]: it for it in CATALOG if isinstance(it, dict) and "id" in it}

def _load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

ROOT = os.path.dirname(__file__)
RULES_DIR = os.path.join(ROOT, "rules")
ANCHOR_THRESHOLDS = _load_json(os.path.join(RULES_DIR, "anchor_thresholds.json"), {})
FEATURE_MULTI_ANCHOR_LOGGING = os.getenv("FEATURE_MULTI_ANCHOR_LOGGING", "0") == "1"

# =========================
# Seed rollback utilities
# =========================
SEED_FILE = os.path.join(HERE, "rules", "seed_triads.json")
SEED_TOGGLE_FILE = os.path.join(HERE, "rules", "seed_toggle.json")

def load_seeds() -> Dict[str, List[str]]:
    seeds = load_json(os.path.relpath(SEED_FILE, HERE), default={})
    return seeds or {}

def _now() -> int:
    return int(time.time())

def seed_mode_status() -> Dict[str, Any]:
    data = load_json(os.path.relpath(SEED_TOGGLE_FILE, HERE), default={"enabled": False, "until": 0})
    # auto-expire
    if data.get("enabled") and _now() >= int(data.get("until", 0)):
        data = {"enabled": False, "until": 0}
        try:
            with open(SEED_TOGGLE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass
    return data

def enable_seed_mode(minutes: int = 60) -> Dict[str, Any]:
    data = {"enabled": True, "until": _now() + max(1, int(minutes))*60}
    os.makedirs(os.path.dirname(SEED_TOGGLE_FILE), exist_ok=True)
    with open(SEED_TOGGLE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data

def disable_seed_mode() -> Dict[str, Any]:
    data = {"enabled": False, "until": 0}
    os.makedirs(os.path.dirname(SEED_TOGGLE_FILE), exist_ok=True)
    with open(SEED_TOGGLE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data

def map_ids_to_output(ids: List[str]) -> Optional[List[Dict[str, Any]]]:
    """Builds output triad from catalog ids. Returns None if invalid/missing data."""
    out: List[Dict[str, Any]] = []
    for pid in ids:
        it = CAT_BY_ID.get(pid)
        if not it: return None
        pal = it.get("palette") or []
        if not isinstance(pal, list) or not pal: return None
        out.append({
            "id": it["id"],
            "title": it["title"],
            "desc": it["desc"],
            "image": it["image_url"],
            "price": it["price_inr"],
            "currency": "INR",
            "emotion": it["emotion"],
            "tier": it["tier"],
            "packaging": it.get("packaging"),
            "mono": bool(it.get("mono")),
            "palette": pal,
            "luxury_grand": bool(it.get("luxury_grand"))
        })
    # Validate 2 MIX + 1 MONO
    if len(out) != 3: return None
    if sum(1 for x in out if x["mono"]) != 1: return None
    ids_set = set()
    for x in out:
        if x["id"] in ids_set: return None
        ids_set.add(x["id"])
    return out

# =========================
# Helpers
# =========================
def sanitize_text(s: str) -> str:
    return " ".join((s or "").strip().split())

def write_golden_artifact(
    request_id: str,
    persona: str,
    prompt: str, # Added prompt
    context: Dict[str, Any],
    meta: Dict[str, Any], # Added meta
    items: List[Dict[str, Any]],
) -> None:
    """Writes a single-line JSON artifact to a date-stamped log file."""
    try:
        log_dir = GOLDEN_ARTIFACTS_DIR
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{datetime.utcnow().strftime('%Y%m%d')}.log")
        
        # Compute suppressed_recent_count robustly
        ps = (context.get("pool_sizes") or context.get("pool_size") or {})
        pre  = (ps.get("pre_suppress")  or {})
        post = (ps.get("post_suppress") or {})
        pre_total  = sum(int(pre.get(k, 0))  for k in ("classic","signature","luxury"))
        post_total = sum(int(post.get(k, 0)) for k in ("classic","signature","luxury"))
        suppressed_recent_count = meta.get("suppressed_recent_count", max(0, pre_total - post_total))

        # Combine context and meta for richer logging
        artifact = {
            "ts": datetime.utcnow().isoformat(),
            "prompt": prompt,
            "request_id": request_id,
            "persona": persona,
            "resolved_anchor": context.get("resolved_anchor"),
            "item_ids": [item.get("id") for item in items],
            "fallback_reason": context.get("fallback_reason"),
            "pool_sizes": context.get("pool_sizes") or context.get("pool_size"), # Use the one already exposed in context
            "edge_type": meta.get("edge_type"),
            "relationship_context": context.get("relationship_context"),
            "prompt_hash": meta.get("prompt_hash"),
            "suppressed_recent_count": int(suppressed_recent_count),
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(artifact) + "\n")
    except Exception as e:
        logger.error(f"Failed to write golden artifact for request_id={request_id}: {e}")

def append_selection_log(items: List[Dict[str, Any]], request_id: str, latency_ms: int, prompt_len: int, path: str = "/api/curate") -> None:
    """Appends one analytics row per curate request."""
    logs_dir = os.path.join(HERE, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    csv_path = os.path.join(logs_dir, "selection_log.csv")
    header = ["ts","request_id","persona","path","latency_ms","prompt_len","detected_emotion","mix_ids","mono_id","tiers","luxury_grand_flags"]
    detected_emotion = items[0].get("emotion") if items else ""
    mix_ids = ";".join([it["id"] for it in items if not it.get("mono")])
    mono_id = next((it["id"] for it in items if it.get("mono")), "")
    tiers = ";".join([it.get("tier","") for it in items])
    lg_flags = ";".join(["true" if CAT_BY_ID.get(it.get("id",""),{}).get("luxury_grand") else "false" for it in items])
    row = [time.strftime("%Y-%m-%dT%H:%M:%S%z"), request_id, PERSONA, path, str(latency_ms), str(prompt_len), detected_emotion, mix_ids, mono_id, tiers, lg_flags]
    need_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(header)
        w.writerow(row)

def analytics_guard_check() -> Dict[str, Any]:
    """Read last 50 rows of selection_log.csv and compute R/L/O share in MIX items."""
    if not ANALYTICS_ENABLED:
        return {"window": 0, "mix_iconic_ratio": None, "alert": False, "message": "Analytics disabled."}
    logs_dir = os.path.join(HERE, "logs")
    csv_path = os.path.join(logs_dir, "selection_log.csv")
    result = {"window": 0, "mix_iconic_ratio": None, "alert": False, "message": ""}
    if not os.path.exists(csv_path): return result
    with open(csv_path, "r", encoding="utf-8") as f: rows = list(csv.reader(f))
    if len(rows) <= 1: return result
    data, total_mix, iconic_mix = rows[1:][-50:], 0, 0
    for r in data:
        try:
            mix_ids = [x for x in (r[7] if len(r) > 7 else "").split(";") if x]
            for mid in mix_ids:
                total_mix += 1
                it = CAT_BY_ID.get(mid)
                if it and any((f.lower() in ICONIC) for f in (it.get("flowers") or [])): iconic_mix += 1
        except Exception: continue
    result["window"] = len(data)
    if total_mix > 0:
        ratio = iconic_mix / float(total_mix)
        result["mix_iconic_ratio"] = ratio
        if ratio < 0.5:
            result["alert"] = True
            result["message"] = f"Iconic MIX share dipped below 50% over the last {len(data)} requests."
    os.makedirs(logs_dir, exist_ok=True)
    with open(os.path.join(logs_dir, "analytics_guard.json"), "w", encoding="utf-8") as f: json.dump(result, f, indent=2)
    if result["alert"]: logger.warning("[GUARD] MIX iconic ratio %.2f < 0.5 over last %s requests", result.get("mix_iconic_ratio", 0.0), result["window"])
    else: logger.info("[GUARD] MIX iconic ratio %s over last %s requests", f"{result.get('mix_iconic_ratio', 0.0):.2f}" if result.get("mix_iconic_ratio") is not None else "n/a", result["window"])
    return result

async def _coerce_prompt(request: Request) -> str:
    try: data = await request.json()
    except Exception: data = None
    if isinstance(data, dict):
        for k in ("prompt", "text", "q", "message", "input"):
            if isinstance(v := data.get(k), str) and v.strip(): return v.strip()
        for v in data.values():
            if isinstance(v, str) and v.strip(): return v.strip()
    try: form = await request.form()
    except Exception: form = None
    if form:
        for k in ("prompt", "text", "q", "message"):
            if k in form and str(form[k]).strip(): return str(form[k]).strip()
    raw = (await request.body() or b"").decode("utf-8", "ignore").strip()
    if raw and not (raw.startswith("{") or raw.startswith("[")): return raw
    for k in ("prompt", "text", "q", "message", "input"):
        if (v := request.query_params.get(k)) and v.strip(): return v.strip()
    return ""

# ---- v1 public allow-list (drop all internals/unknowns) ----
PUBLIC_FIELDS = {
    "id","title","desc","image","price","currency","emotion",
    "tier","mono","palette","note","edge_case","edge_type"
}
def _sanitize_item(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if k in PUBLIC_FIELDS}
# ------------------------------------------------------------

# =========================
# Schemas (UI-aligned)
# =========================
class CurateContext(BaseModel):
    budget_inr: Optional[int] = None
    emotion_hint: Optional[str] = None
    packaging_pref: Optional[str] = None
    locale: Optional[str] = None
    recent_ids: Optional[List[str]] = None
    session_id: Optional[str] = None
    run_count: Optional[int] = 0

class CurateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500)
    context: Optional[CurateContext] = None

class SeedModeIn(BaseModel):
    minutes: Optional[int] = 60

class CheckoutIn(BaseModel):
    product_id: str

class ItemOut(BaseModel):
    id: str
    title: str
    desc: str
    image: str
    price: int
    currency: str
    emotion: str
    tier: str
    mono: bool
    palette: List[str]
    # optional extras (public-safe)
    edge_case: Optional[bool] = False
    edge_type: Optional[str] = None
    note: Optional[str] = None

# =========================
# Routes
# =========================
@app.get("/health")
def health(): return {"status": "ok", "persona": ERROR_PERSONA, "version": "v1"}

@app.get("/api/curate/seed_mode")
def seed_status(): return seed_mode_status()

@app.post("/api/curate/seed_mode")
def seed_enable(body: SeedModeIn): return enable_seed_mode(max(1, int(body.minutes or 60)))

@app.post("/api/curate/seed_mode/disable")
def seed_disable(): return disable_seed_mode()

@app.post("/api/curate", summary="Curate", response_model=List[ItemOut])
@limiter.limit(f"{RATE_LIMIT_PER_MIN}/minute")
async def curate_post(body: CurateRequest, request: Request, response: Response):
    """JSON-only canonical endpoint. Returns items validated against the ItemOut schema."""
    started = time.time()
    request_id = str(uuid.uuid4())
    prompt = body.prompt.strip()
    req_context = body.context.dict() if isinstance(body.context, CurateContext) else {}
    req_context["request_id"] = request_id

    try:
        final_triad, context, meta = selection_engine(prompt=prompt, context=req_context)
        items = _transform_for_api(final_triad, context.get("resolved_anchor"))
        items = [_sanitize_item(it) for it in items]
    except Exception as e:
        logger.error(f"Engine error for request_id={request_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": {"code": "ENGINE_ERROR", "message": str(e)[:400]}})

    latency_ms = int((time.time() - started) * 1000)
    logger.info("[%s] CURATE ip=%s emotion=%s latency_ms=%s prompt_len=%s",
                 PERSONA, get_remote_address(request), items[0].get("emotion",""), latency_ms, len(prompt))

    append_selection_log(items, request_id, latency_ms, len(prompt), path="/api/curate")
    analytics_guard_check()
    # Pass prompt, context, and meta to the artifact writer
    write_golden_artifact(request_id, PERSONA, prompt, context, meta, items)

    emit_header = ANCHOR_THRESHOLDS.get("multi_anchor_logging", {}).get("emit_header", False)
    if FEATURE_MULTI_ANCHOR_LOGGING and emit_header:
        # Re-detect emotion to get scores (avoid passing scores back from engine)
        _, _, scores = detect_emotion(normalize(prompt), context or {})
        if scores:
            top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            s1 = round(top[0][1], 2)
            header_vals = [f"{top[0][0]}:{s1}"]
            if len(top) > 1:
                th = ANCHOR_THRESHOLDS.get("multi_anchor_logging", {})
                s2 = round(top[1][1], 2)
                # Check if second score is high enough and close enough to the first
                if s2 >= float(th.get("score2_min", 0.25)) and (s1 - s2) <= float(th.get("delta_max", 0.15)):
                    header_vals.append(f"{top[1][0]}:{s2}")
            response.headers["X-Detected-Emotions"] = ",".join(header_vals)

    return items

# =========================
# Flexible/Legacy Routes
# =========================
async def _curate_flexible(request: Request) -> JSONResponse:
    """Coercion shim for legacy/alias routes."""
    started = time.time()
    request_id = str(uuid.uuid4())
    prompt = await _coerce_prompt(request)
    if not prompt:
        return error_json("PROMPT_REQUIRED", "Provide a non-empty 'prompt'.", 400, request_id)

    prompt = sanitize_text(prompt)
    req_context = {"request_id": request_id}

    try:
        # --> This line already gets the meta object correctly
        final_triad, context, meta = selection_engine(prompt=prompt, context=req_context)
        items = _transform_for_api(final_triad, context.get("resolved_anchor"))
        items = [_sanitize_item(it) for it in items]
    except Exception as e:
        logger.error(f"Engine error for request_id={request_id} (shim): {e}", exc_info=True)
        return error_json("ENGINE_ERROR", str(e)[:400], 500, request_id)

    latency_ms = int((time.time() - started) * 1000)
    logger.info("[%s] CURATE(SHIM) ip=%s emotion=%s latency_ms=%s prompt_len=%s",
                 PERSONA, get_remote_address(request), items[0].get("emotion",""), latency_ms, len(prompt))
    append_selection_log(items, request_id, latency_ms, len(prompt), path="/curate(shim)")
    analytics_guard_check()
    # Pass prompt, context, and the ACTUAL meta object to the artifact writer
    # --- THIS IS THE FIX ---
    write_golden_artifact(request_id, PERSONA, prompt, context, meta, items) # Was passing {} for meta

    return JSONResponse(items)

@app.get("/api/curate")
@limiter.limit(f"{RATE_LIMIT_PER_MIN}/minute")
async def curate_get(request: Request):
    started = time.time()
    request_id = str(uuid.uuid4())
    prompt = await _coerce_prompt(request)
    if not isinstance(prompt, str) or len(prompt.strip()) < 3:
        return JSONResponse({"error": "Invalid input."}, status_code=400)

    prompt = sanitize_text(prompt)
    if not prompt:
        return error_json("EMPTY_PROMPT", "Please write a short line.", 422, request_id)

    req_context = {"request_id": request_id}

    seed_state = seed_mode_status()
    items = None
    context = {}
    meta = {} # Initialize meta for GET route

    if seed_state.get("enabled"):
        norm = normalize(prompt)
        emo, _, _ = detect_emotion(norm, req_context)
        seeds = load_seeds().get(emo) or []
        if len(seeds) == 3:
            seeded_items = map_ids_to_output(seeds)
            if seeded_items:
                items = _transform_for_api(seeded_items, emo)
                items = [_sanitize_item(it) for it in items]
                # Populate minimal context and meta for seed mode logging
                context = {"resolved_anchor": emo, "request_id": request_id}
                meta = {"resolved_anchor": emo} # Mimic some meta fields

    if items is None: # If not in seed mode or seed failed
        try:
            final_triad, context, meta = selection_engine(prompt=prompt, context=req_context)
            items = _transform_for_api(final_triad, context.get("resolved_anchor"))
            items = [_sanitize_item(it) for it in items]
        except Exception as e:
            logger.error(f"Engine error for request_id={request_id} (GET): {e}", exc_info=True)
            return error_json("ENGINE_ERROR", str(e)[:400], 500, request_id)

    latency_ms = int((time.time() - started) * 1000)
    logger.info("[%s] CURATE(GET) ip=%s emotion=%s latency_ms=%s prompt_len=%s%s",
                 PERSONA, get_remote_address(request), items[0].get("emotion",""), latency_ms, len(prompt),
                 " (seed-mode)" if seed_state.get("enabled") and items is not None else "")

    append_selection_log(items, request_id, latency_ms, len(prompt), path="/api/curate_get")
    analytics_guard_check()
    # Pass prompt, context, and meta (potentially empty/minimal for GET/seed) to the artifact writer
    write_golden_artifact(request_id, PERSONA, prompt, context, meta, items)

    return JSONResponse(items)

@app.post("/curate")
@limiter.limit(f"{RATE_LIMIT_PER_MIN}/minute")
async def curate_alias_post(request: Request): return await _curate_flexible(request)

@app.get("/curate")
@limiter.limit(f"{RATE_LIMIT_PER_MIN}/minute")
async def curate_alias_get(request: Request): return await curate_get(request)

@app.post("/api/curate/next", summary="Curate (rotate next)", response_model=List[ItemOut])
async def curate_post_next(body: CurateRequest, request: Request):
    """Gets the next set of items, ensuring the response shape is identical to the primary endpoint."""
    request_id, req_context = str(uuid.uuid4()), body.context.dict() if isinstance(body.context, CurateContext) else {}
    req_context["run_count"] = int(req_context.get("run_count") or 0) + 1
    req_context["request_id"] = request_id
    try:
        final_triad, context, meta = selection_engine(prompt=body.prompt.strip(), context=req_context)
        items = _transform_for_api(final_triad, context.get("resolved_anchor"))
        items = [_sanitize_item(it) for it in items]
    except Exception as e:
        logger.error(f"Engine error for request_id={request_id} (next): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": {"code": "ENGINE_ERROR", "message": str(e)[:400]}})
    # Note: /next endpoint does not write golden artifacts or standard logs currently
    return items

# -------------------------
# Golden-Set Harness (Internal Test Tool)
# -------------------------
GOLDEN_TESTS: List[Dict[str, Any]] = [
    {"name": "romance_budget_2000", "prompt": "romantic anniversary under 2000", "context": {"budget_inr": 2000}},
    {"name": "romance_plain", "prompt": "romantic bouquet please"},
    {"name": "only_lilies", "prompt": "only lilies please", "expect_lily_mono": True},
    {"name": "hydrangea_redirect", "prompt": "hydrangea bouquet", "expect_note": True},
    {"name": "celebration_bright", "prompt": "bright congratulations"},
    {"name": "encouragement_exams", "prompt": "encouragement for exams"},
    {"name": "gratitude_thanks", "prompt": "thank you flowers"},
    {"name": "friendship_care", "prompt": "for a dear friend"},
    {"name": "encouragement_getwell", "prompt": "get well soon flowers"},
    {"name": "birthday", "prompt": "birthday flowers"},
    {"name": "sympathy_loss", "prompt": "i’m so sorry for your loss"},
    {"name": "apology", "prompt": "i deeply apologize"},
    {"name": "farewell", "prompt": "farewell flowers"},
    {"name": "valentine", "prompt": "valentine surprise"}
]

def _run_one(test: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"name": test["name"], "prompt": test["prompt"], "status": "PASS", "reasons": []}
    try:
        # Pass request_id for better test traceability
        test_context = test.get("context") or {}
        test_context["request_id"] = f"golden-harness-{uuid.uuid4().hex}"
        items, _, _ = selection_engine(prompt=test["prompt"], context=test_context)
    except Exception as e:
        out["status"] = "FAIL"
        out["reasons"].append(f"engine_error: {repr(e)}")
        return out

    # Basic structural checks
    if len(items) != 3: out["status"] = "FAIL"; out["reasons"].append("triad_len != 3")
    if sum(1 for it in items if it.get("mono")) != 1: out["status"] = "FAIL"; out["reasons"].append("mono_count != 1")
    if any(not isinstance(it.get("palette"), list) or len(it["palette"]) == 0 for it in items): out["status"] = "FAIL"; out["reasons"].append("palette missing")
    if sum(1 for it in items if it.get("luxury_grand")) > 1: out["status"] = "FAIL"; out["reasons"].append(">1 luxury_grand")

    # Specific test expectations
    if test.get("expect_lily_mono"):
        mono_item = next((it for it in items if it.get("mono")), None)
        if not mono_item: out["status"] = "FAIL"; out["reasons"].append("no mono item")
        else:
            cat = CAT_BY_ID.get(mono_item["id"]) or {}
            if "lily" not in [s.lower() for s in (cat.get("flowers") or [])]: out["status"] = "FAIL"; out["reasons"].append("mono is not lily")
    if test.get("expect_note") and not any(it.get("note") for it in items): out["status"] = "FAIL"; out["reasons"].append("redirection note missing")

    # Record results
    out["emotion"] = items[0].get("emotion", ""); out["ids"] = [it.get("id") for it in items]
    out["tiers"] = [it.get("tier") for it in items]; out["mono_id"] = next((it.get("id") for it in items if it.get("mono")), "")
    return out

@app.post("/api/curate/golden")
def curate_golden(request: Request):
    started = time.time()
    results = [_run_one(t) for t in GOLDEN_TESTS]
    passed = sum(1 for r in results if r["status"] == "PASS")
    summary = {
        "ts": datetime.utcnow().isoformat(), "persona": ERROR_PERSONA, "total": len(results),
        "passed": passed, "failed": len(results) - passed,
        "latency_ms": int((time.time() - started) * 1000), "results": results
    }
    try:
        logs_dir = os.path.join(HERE, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        with open(os.path.join(logs_dir, "golden_harness_runs.log"), "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
    except Exception as e:
        logger.error(f"Failed to write golden harness artifact: {e}")
    return summary

@app.get("/api/curate/golden/p1")
def curate_golden_p1(request: Request):
    """Alias for Phase-1 golden suite."""
    return curate_golden(request)

@app.post("/api/checkout")
@limiter.limit(f"{RATE_LIMIT_PER_MIN}/minute")
def checkout(body: CheckoutIn, request: Request):
    pid = (body.product_id or "").strip()
    if not pid: return error_json("BAD_PRODUCT", "product_id is required.", 422)
    url = f"https://checkout.example/intent?pid={pid}"
    logger.info(f"[{PERSONA}] CHECKOUT ip={get_remote_address(request)} product={pid}")
    return {"checkout_url": url}

# =========================
# Error Normalization
# =========================
def error_json(code: str, message: str, status: int = 400, request_id: Optional[str] = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}, "persona": ERROR_PERSONA, "request_id": request_id or str(uuid.uuid4())})

@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict):
        shaped = {"error": exc.detail.get("error", {"code": "HTTP_ERROR", "message": "Request error."}), "persona": ERROR_PERSONA, "request_id": str(uuid.uuid4())}
        return JSONResponse(status_code=exc.status_code, content=shaped)
    return await http_exception_handler(request, exc)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc): return error_json("VALIDATION_ERROR", "Invalid input.", 422)

@app.exception_handler(RateLimitExceeded)
async def ratelimit_handler(request: Request, exc: RateLimitExceeded): return error_json("RATE_LIMITED", "Too many requests. Please try again in a minute.", 429)
