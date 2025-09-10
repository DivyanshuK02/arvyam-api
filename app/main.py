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

# Selection Engine (Step C wiring)
from selection_engine import selection_engine

# =========================
# Environment & Constants
# =========================
ALLOWED = os.getenv("ALLOWED_ORIGINS", "https://arvyam.com")
ASSET_BASE = os.getenv("PUBLIC_ASSET_BASE", "https://arvyam.com").rstrip("/")
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "10"))
PERSONA = os.getenv("PERSONA_NAME", "ARVY")  # for logs/UI
ERROR_PERSONA = "ARVY"                       # hard-coded in API errors

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
# Data Loaders (legacy compatibility)
# =========================
HERE = os.path.dirname(__file__)

def load_json(relpath: str):
    with open(os.path.join(HERE, relpath), "r", encoding="utf-8") as f:
        return json.load(f)

# Legacy placeholders (not used by /api/curate anymore)
# Kept to avoid breaking any other parts of your app that might still import these.
try:
    RULES: List[Dict[str, Any]] = load_json("rules.json")
except Exception:
    RULES = []
try:
    CATALOG: List[Dict[str, Any]] = load_json("catalog.json")
except Exception:
    CATALOG = []

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
    # derive simple features from items
    detected_emotion = items[0].get("emotion") if items else ""
    mix_ids = ";".join([it["id"] for it in items if not it.get("mono")])
    mono_id = next((it["id"] for it in items if it.get("mono")), "")
    tiers = ";".join([it.get("tier","") for it in items])
    lg_flags = ";".join(["true" if it.get("luxury_grand") else "false" for it in items])
    row = [time.strftime("%Y-%m-%dT%H:%M:%S%z"), request_id, PERSONA, path, str(latency_ms), str(prompt_len), detected_emotion, mix_ids, mono_id, tiers, lg_flags]
    # write header if file doesn't exist
    need_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(header)
        w.writerow(row)

# =========================
# Canonical Error Builder
# =========================
def error_json(code: str, message: str, status: int = 400) -> JSONResponse:
    """Single place to shape all error responses with persona + request_id."""
    return JSONResponse(
        status_code=status,
        content={
            "error": {"code": code, "message": message},
            "persona": ERROR_PERSONA,           # always "ARVY"
            "request_id": str(uuid.uuid4())
        }
    )

# =========================
# Schemas (UI-aligned)
# =========================
class CurateIn(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500)
    # Optional beginner-safe context
    context: Optional[Dict[str, Any]] = None  # keys: emotion_hint, budget_inr, packaging_pref, locale

class CheckoutIn(BaseModel):
    product_id: str

# =========================
# Routes
# =========================
@app.get("/health")
def health():
    return {"status": "ok", "persona": ERROR_PERSONA, "version": "v1"}

@app.post("/api/curate")
@limiter.limit(f"{RATE_LIMIT_PER_MIN}/minute")
def curate(body: CurateIn, request: Request):
    started = time.time()
    try:
        prompt = sanitize_text(body.prompt)
        if not prompt:
            return error_json("EMPTY_PROMPT", "Please write a short line.", 422)
        prompt_len = len(prompt)

        # Run Selection Engine (returns exactly 3 items or raises)
        items = selection_engine(prompt=prompt, context=body.context or {})

        # Basic metrics and a request id
        latency_ms = int((time.time() - started) * 1000)
        request_id = str(uuid.uuid4())

        # Safe logs (never log full prompt)
        ip = get_remote_address(request)
        logging.info("[%s] CURATE ip=%s emotion=%s latency_ms=%s prompt_len=%s", PERSONA, ip, items[0].get("emotion",""), latency_ms, prompt_len)

        # CSV analytics guard (R/L/O share checks can be done offline)
        append_selection_log(items, request_id, latency_ms, prompt_len)

        # Return just the 3 items (keeps the UI stable)
        return items

    except Exception as e:
        logging.exception("[%s] CURATE_ERROR %s", PERSONA, repr(e))
        return error_json("CURATE_ERROR", "We couldn’t curate this request. Please try a different phrase.", 400)

@app.post("/api/checkout")
@limiter.limit(f"{RATE_LIMIT_PER_MIN}/minute")
def checkout(body: CheckoutIn, request: Request):
    pid = sanitize_text(body.product_id)
    if not pid:
        return error_json("BAD_PRODUCT", "product_id is required.", 422)
    url = f"https://checkout.example/intent?pid={pid}"
    ip = get_remote_address(request)
    logger.info(f"[{PERSONA}] CHECKOUT ip=%s product=%s", ip, pid)
    return {"checkout_url": url}

# =========================
# Error Normalization
# =========================
@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    # Normalize any legacy raises into our canonical shape
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