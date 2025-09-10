import os, json, logging, uuid, time, csv
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
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

# Selection Engine and helpers
from selection_engine import selection_engine, normalize, detect_emotion

# =========================
# Environment & Constants
# =========================
ALLOWED = os.getenv("ALLOWED_ORIGINS", "https://arvyam.com")
ASSET_BASE = os.getenv("PUBLIC_ASSET_BASE", "https://arvyam.com").rstrip("/")
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "10"))
PERSONA = os.getenv("PERSONA_NAME", "ARVY")  # for logs/UI
ERROR_PERSONA = "ARVY"                       # hard-coded in API errors
ICONIC = {"rose","lily","orchid"}            # for analytics guard

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
            "packaging": it["packaging"],
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

def append_selection_log(items: List[Dict[str, Any]], request_id: str, latency_ms: int, prompt_len: int, path: str = "/api/curate") -> None:
    """
    Appends one analytics row per curate request.
    Schema: ts, request_id, persona, path, latency_ms, prompt_len, detected_emotion, mix_ids, mono_id, tiers, luxury_grand_flags
    """
    logs_dir = os.path.join(HERE, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    csv_path = os.path.join(logs_dir, "selection_log.csv")
    header = ["ts","request_id","persona","path","latency_ms","prompt_len","detected_emotion","mix_ids","mono_id","tiers","luxury_grand_flags"]
    detected_emotion = items[0].get("emotion") if items else ""
    mix_ids = ";".join([it["id"] for it in items if not it.get("mono")])
    mono_id = next((it["id"] for it in items if it.get("mono")), "")
    tiers = ";".join([it.get("tier","") for it in items])
    lg_flags = ";".join(["true" if it.get("luxury_grand") else "false" for it in items])
    row = [time.strftime("%Y-%m-%dT%H:%M:%S%z"), request_id, PERSONA, path, str(latency_ms), str(prompt_len), detected_emotion, mix_ids, mono_id, tiers, lg_flags]

    need_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(header)
        w.writerow(row)

def analytics_guard_check() -> Dict[str, Any]:
    """Read last 50 rows of selection_log.csv and compute R/L/O share in MIX items."""
    logs_dir = os.path.join(HERE, "logs")
    csv_path = os.path.join(logs_dir, "selection_log.csv")
    result = {"window": 0, "mix_iconic_ratio": None, "alert": False, "message": ""}
    if not os.path.exists(csv_path):
        return result
    # Load lines (skip header)
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) <= 1:
        return result
    data = rows[1:][-50:]  # last up to 50 rows
    total_mix = 0
    iconic_mix = 0
    for r in data:
        try:
            # columns: ts, request_id, persona, path, latency_ms, prompt_len, detected_emotion, mix_ids, mono_id, tiers, lg
            mix_ids_field = r[7] if len(r) > 7 else ""
            mix_ids = [x for x in mix_ids_field.split(";") if x]
            for mid in mix_ids:
                total_mix += 1
                it = CAT_BY_ID.get(mid)
                flowers = (it.get("flowers") or []) if it else []
                if any((f.lower() in ICONIC) for f in flowers):
                    iconic_mix += 1
        except Exception:
            continue
    result["window"] = len(data)
    if total_mix > 0:
        ratio = iconic_mix / float(total_mix)
        result["mix_iconic_ratio"] = ratio
        if ratio < 0.5:
            result["alert"] = True
            result["message"] = "Iconic MIX share dipped below 50% over the last {} requests.".format(len(data))
    # Persist guard status
    os.makedirs(logs_dir, exist_ok=True)
    with open(os.path.join(logs_dir, "analytics_guard.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    if result["alert"]:
        logger.warning("[GUARD] MIX iconic ratio %.2f < 0.5 over last %s requests", result.get("mix_iconic_ratio", 0.0), result["window"])
    else:
        logger.info("[GUARD] MIX iconic ratio %.2f over last %s requests", result.get("mix_iconic_ratio", 0.0) if result["mix_iconic_ratio"] is not None else -1, result["window"])
    return result

# =========================
# Schemas (UI-aligned)
# =========================
class CurateIn(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500)
    context: Optional[Dict[str, Any]] = None  # emotion_hint, budget_inr, packaging_pref, locale

class SeedModeIn(BaseModel):
    minutes: Optional[int] = 60

class CheckoutIn(BaseModel):
    product_id: str

# =========================
# Routes
# =========================
@app.get("/health")
def health():
    return {"status": "ok", "persona": ERROR_PERSONA, "version": "v1"}

@app.get("/api/curate/seed_mode")
def seed_status():
    return seed_mode_status()

@app.post("/api/curate/seed_mode")
def seed_enable(body: SeedModeIn):
    data = enable_seed_mode(max(1, int(body.minutes or 60)))
    return data

@app.post("/api/curate/seed_mode/disable")
def seed_disable():
    return disable_seed_mode()

@app.post("/api/curate")
@limiter.limit(f"{RATE_LIMIT_PER_MIN}/minute")
def curate(body: CurateIn, request: Request):
    started = time.time()
    try:
        prompt = sanitize_text(body.prompt)
        if not prompt:
            return error_json("EMPTY_PROMPT", "Please write a short line.", 422)
        prompt_len = len(prompt)

        # Seed-mode check (1 hour window)
        seed_state = seed_mode_status()
        items = None
        if seed_state.get("enabled"):
            # Determine emotion using the engine's normalize+detect
            norm = normalize(prompt)
            emo = detect_emotion(norm, body.context or {})
            seeds = load_seeds().get(emo) or []
            if len(seeds) == 3:
                seeded = map_ids_to_output(seeds)
                if seeded:
                    items = seeded

        # If no seed triad (or invalid), fall back to engine
        if items is None:
            items = selection_engine(prompt=prompt, context=body.context or {})

        # Basic metrics and a request id
        latency_ms = int((time.time() - started) * 1000)
        request_id = str(uuid.uuid4())

        # Safe logs (never log full prompt)
        ip = get_remote_address(request)
        logging.info("[%s] CURATE ip=%s emotion=%s latency_ms=%s prompt_len=%s%s",
                     PERSONA, ip, items[0].get("emotion",""), latency_ms, prompt_len,
                     " (seed-mode)" if seed_state.get("enabled") and items is not None else "")

        # CSV analytics guard (R/L/O share checks offline)
        append_selection_log(items, request_id, latency_ms, prompt_len)
        analytics_guard_check()

        return items

    except Exception as e:
        logging.exception("[%s] CURATE_ERROR %s", PERSONA, repr(e))
        return error_json("CURATE_ERROR", "We couldn’t curate this request. Please try a different phrase.", 400)

@app.post("/api/checkout")
@limiter.limit(f"{RATE_LIMIT_PER_MIN}/minute")
def checkout(body: CheckoutIn, request: Request):
    pid = (body.product_id or "").strip()
    if not pid:
        return error_json("BAD_PRODUCT", "product_id is required.", 422)
    url = f"https://checkout.example/intent?pid={pid}"
    ip = get_remote_address(request)
    logger.info(f"[{PERSONA}] CHECKOUT ip=%s product=%s", ip, pid)
    return {"checkout_url": url}

# =========================
# Error Normalization
# =========================
def error_json(code: str, message: str, status: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {"code": code, "message": message},
            "persona": ERROR_PERSONA,
            "request_id": str(uuid.uuid4())
        }
    )

@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict):
        shaped = {
            "error": exc.detail.get("error", {"code": "HTTP_ERROR", "message": "Request error."}),
            "persona": ERROR_PERSONA,
            "request_id": str(uuid.uuid4())
        }
        return JSONResponse(status_code=exc.status_code, content=shaped)
    return await http_exception_handler(request, exc)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc):
    return error_json("VALIDATION_ERROR", "Invalid input.", 422)

@app.exception_handler(RateLimitExceeded)
async def ratelimit_handler(request: Request, exc: RateLimitExceeded):
    return error_json("RATE_LIMITED", "Too many requests. Please try again in a minute.", 429)