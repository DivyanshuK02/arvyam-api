
import os, json, logging
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError

# Rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded

# ---------- Env ----------
ALLOWED = os.getenv("ALLOWED_ORIGINS", "https://arvyam.com")
ASSET_BASE = os.getenv("PUBLIC_ASSET_BASE", "https://arvyam.com").rstrip("/")
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "10"))

# ---------- Logging ----------
logger = logging.getLogger("arvyam")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------- App ----------
limiter = Limiter(key_func=get_remote_address, default_limits=[f"{RATE_LIMIT_PER_MIN}/minute"])
app = FastAPI(title="Arvyam API (MVP Neutral Titles)")
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED.split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Data ----------
HERE = os.path.dirname(__file__)

def load_json(relpath: str):
    with open(os.path.join(HERE, relpath), "r", encoding="utf-8") as f:
        return json.load(f)

RULES: List[Dict[str, Any]] = load_json("rules.json")
CATALOG: List[Dict[str, Any]] = load_json("catalog.json")

# ---------- Helpers ----------
def to_public_image(url_or_path: str) -> str:
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return url_or_path
    return f"{ASSET_BASE}{url_or_path}" if ASSET_BASE else url_or_path

def sanitize_text(s: str) -> str:
    return " ".join((s or "").strip().split())

def classify_category(prompt: str) -> str:
    p = prompt.lower()
    best_cat = None
    best_hits = 0
    for r in RULES:
        kws = [k.lower() for k in (r.get("keywords") or [])]
        hits = sum(1 for k in kws if k and k in p)
        if hits > best_hits:
            best_cat = r["category"]
            best_hits = hits
    if best_cat:
        return best_cat
    for fallback in ["Love", "Encouragement", "Gratitude"]:
        if any(r["category"].lower() == fallback.lower() for r in RULES):
            return fallback
    return RULES[0]["category"]

def shape_item(r: Dict[str, Any]) -> Dict[str, Any]:
    # Force a neutral base title so we don't repeat 'Rose Bouquet' etc.
    return {
        "id": r["id"],
        "title": "Curated Bouquet",        # <— neutral base title
        "desc": r.get("desc") or "Thoughtfully arranged.",
        "image": to_public_image(r.get("image_url") or r.get("image","")),
        "price": int(r["price_inr"]),
        "currency": "INR",
    }

def pick_three(category: str) -> List[Dict[str, Any]]:
    candidates = [c for c in CATALOG if category in (c.get("tags") or [])]
    if not candidates:
        candidates = CATALOG[:]

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
                img = (c.get("image_url") or c.get("image",""))
                if img not in seen_imgs:
                    seen_imgs.add(img); return c
        return None

    for p in targets:
        item = first_at(p)
        if item and item not in chosen:
            chosen.append(item)

    for c in candidates:
        if len(chosen) >= 3: break
        img = (c.get("image_url") or c.get("image",""))
        if img not in seen_imgs and c not in chosen:
            seen_imgs.add(img); chosen.append(c)

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
        base_title = "Curated Bouquet"   # <— fixed neutral base
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

# ---------- Schemas ----------
class CurateIn(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=350)

class CheckoutIn(BaseModel):
    product_id: str

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/curate")
@limiter.limit(f"{RATE_LIMIT_PER_MIN}/minute")
def curate(body: CurateIn, request: Request):
    try:
        prompt = sanitize_text(body.prompt)
        if not prompt:
            raise HTTPException(status_code=422, detail={"error":{"code":"EMPTY_PROMPT","message":"Please write a short line."}})

        category = classify_category(prompt)
        items = pick_three(category)
        items = refine_copy(items, category)

        ip = get_remote_address(request)
        logger.info("CURATE ip=%s category=%s prompt=%s", ip, category, prompt[:120])
        return items

    except HTTPException:
        raise
    except Exception:
        logger.exception("CURATE_ERROR")
        raise HTTPException(status_code=500, detail={"error":{"code":"SERVER_ERROR","message":"Something went wrong while curating. Please try again later."}})

@app.post("/api/checkout")
@limiter.limit(f"{RATE_LIMIT_PER_MIN}/minute")
def checkout(body: CheckoutIn, request: Request):
    pid = sanitize_text(body.product_id)
    if not pid:
        raise HTTPException(status_code=422, detail={"error":{"code":"BAD_PRODUCT","message":"product_id is required."}})
    url = f"https://checkout.example/intent?pid={pid}"
    ip = get_remote_address(request)
    logger.info("CHECKOUT ip=%s product=%s", ip, pid)
    return {"checkout_url": url}

# ---------- Error shaping ----------
@app.exception_handler(HTTPException)
async def http_exc_handler(request, exc: HTTPException):
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return await http_exception_handler(request, exc)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"error":{"code":"VALIDATION_ERROR","message":"Invalid input."}}
    )

@app.exception_handler(RateLimitExceeded)
async def ratelimit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error":{"code":"RATE_LIMITED","message":"Too many requests. Please try again in a minute."}}
    )
