import os, json, logging, uuid
from typing import Any, Dict, List

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
# Data Loaders
# =========================
HERE = os.path.dirname(__file__)

def load_json(relpath: str):
    with open(os.path.join(HERE, relpath), "r", encoding="utf-8") as f:
        return json.load(f)

RULES: List[Dict[str, Any]] = load_json("rules.json")
CATALOG: List[Dict[str, Any]] = load_json("catalog.json")

# =========================
# Helpers
# =========================
def to_public_image(url_or_path: str) -> str:
    if url_or_path.startswith(("http://", "https://")):
        return url_or_path
    return f"{ASSET_BASE}{url_or_path}" if ASSET_BASE else url_or_path

def sanitize_text(s: str) -> str:
    return " ".join((s or "").strip().split())

def classify_category(prompt: str) -> str:
    p = prompt.lower()
    best_cat, best_hits = None, 0
    for r in RULES:
        kws = [k.lower() for k in (r.get("keywords") or [])]
        hits = sum(1 for k in kws if k and k in p)
        if hits > best_hits:
            best_cat, best_hits = r["category"], hits
    if best_cat:
        return best_cat
    # fallback order
    for fallback in ["Love", "Encouragement", "Gratitude"]:
        if any(r["category"].lower() == fallback.lower() for r in RULES):
            return fallback
    return RULES[0]["category"]

def shape_item(r: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": r["id"],
        "title": "Curated Bouquet",
        "desc": r.get("desc") or "Thoughtfully arranged.",
        "image": to_public_image(r.get("image_url") or r.get("image", "")),
        "price": int(r["price_inr"]),
        "currency": "INR",
    }

def pick_three(category: str) -> List[Dict[str, Any]]:
    candidates = [c for c in CATALOG if category in (c.get("tags") or [])] or CATALOG[:]
    candidates.sort(key=lambda x: x.get("price_inr", 999999))

    prices = sorted({c.get("price_inr", 0) for c in candidates})
    if len(prices) >= 3:
        targets = [prices[0], prices[len(prices)//2], prices[-1]]
    elif len(prices) == 2:
        targets = [prices[0], prices[0], prices[1]]
    else:
        targets = [prices[0], prices[0], prices[0]]

    chosen, seen_imgs = [], set()

    def first_at(price):
        for c in candidates:
            if c.get("price_inr") == price:
                img = (c.get("image_url") or c.get("image", ""))
                if img not in seen_imgs:
                    seen_imgs.add(img)
                    return c
        return None

    for p in targets:
        item = first_at(p)
        if item and item not in chosen:
            chosen.append(item)

    for c in candidates:
        if len(chosen) >= 3:
            break
        img = (c.get("image_url") or c.get("image", ""))
        if img not in seen_imgs and c not in chosen:
            seen_imgs.add(img)
            chosen.append(c)

    i = 0
    while len(chosen) < 3 and i < len(candidates):
        if candidates[i] not in chosen:
            chosen.append(candidates[i])
        i += 1

    return [shape_item(x) for x in chosen[:3]]

def refine_copy(items: List[Dict[str, Any]], category: str) -> List[Dict[str, Any]]:
    prices = sorted({it["price"] for it in items})
    if len(prices) >= 3:
        low, mid, high = prices[0], prices[len(prices)//2], prices[-1]
    elif len(prices) == 2:
        low, mid, high = prices[0], prices[0], prices[1]
    else:
        low = mid = high = prices[0]

    tones = {
        "love": ("A simple, heartfelt gesture.", "Thoughtful and softly romantic.", "Luxe and timeless."),
        "encouragement": ("A warm lift of spirit.", "Uplifting with gentle accents.", "A fuller, elegant pick-me-up."),
        "gratitude": ("A kind thank-you.", "Warm appreciation, softly styled.", "A gracious, elevated thank-you."),
        "sympathy": ("Calm and respectful.", "Peaceful, light arrangement.", "Serene, composed statement."),
        "celebration": ("Bright and cheerful.", "Festive with soft greens.", "Lush and celebratory."),
    }
    key = next((k for k in tones if k.lower() in category.lower()), None)
    low_t, mid_t, high_t = tones.get(key, ("A simple gesture.", "Thoughtful and balanced.", "Luxe and refined."))

    for it in items:
        p = it["price"]
        base_title = "Curated Bouquet"
        if p == low:
            it["title"] = f"{base_title} • Classic"
            it["desc"] = low_t
        elif p == mid:
            it["title"] = f"{base_title} • Signature"
            it["desc"] = mid_t
        elif p == high:
            it["title"] = f"{base_title} • Luxury"
            it["desc"] = high_t
    return items

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
# Schemas (UI-aligned caps)
# =========================
class CurateIn(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=240)  # 240 cap

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
    try:
        prompt = sanitize_text(body.prompt)
        if not prompt:
            return error_json("EMPTY_PROMPT", "Please write a short line.", 422)

        category = classify_category(prompt)
        items = refine_copy(pick_three(category), category)

        ip = get_remote_address(request)
        # Safe, short preview; never log full prompt
        logger.info(f"[{PERSONA}] CURATE ip=%s category=%s preview=%s", ip, category, prompt[:60])
        return items

    except Exception as e:
        logger.exception(f"[{PERSONA}] CURATE_ERROR")
        return error_json("SERVER_ERROR", "Something went wrong while curating. Please try again later.", 500)

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
