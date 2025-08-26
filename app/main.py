
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
app = FastAPI(title="Arvyam API (MVP)")
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

# From Step 3 bundle (keep these files in app/)
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
    """Match the user prompt to the best category from rules.json by keyword hits.
    Fallback to Love/Encouragement/Gratitude if no hits, else first rule."""
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

def flower_priority_for_category(category: str) -> List[str]:
    """Return prioritized flowers for a category, boosting roses/lilies if present."""
    for r in RULES:
        if r["category"].lower() == category.lower():
            pr = [x.title() for x in (r.get("priority") or [])]
            def boost(arr, word):
                arr2 = [x for x in arr if x.lower() != word]
                if any(x.lower() == word for x in arr):
                    arr2 = [word.title()] + arr2
                return arr2
            pr = boost(pr, "rose"); pr = boost(pr, "roses")
            pr = boost(pr, "lily"); pr = boost(pr, "lilies")
            return pr
    return []

def shape_item(r: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": r["id"],
        "title": r["title"],
        "desc": r["desc"],
        "image": to_public_image(r.get("image_url") or r.get("image","")),
        "price": int(r["price_inr"]),
        "currency": "INR",
    }

def pick_three(category: str) -> List[Dict[str, Any]]:
    """Select exactly 3 items for the category:
    - Filter by category tag
    - Sort by flower priority (roses/lilies boosted) then price
    - Aim for low/mid/high prices
    - Enforce distinct images per response (best-effort)"""
    # 1) filter candidates
    candidates = [c for c in CATALOG if category in (c.get("tags") or [])]

    # 2) sort by flower priority then by price
    prio = flower_priority_for_category(category)
    def prio_index(flower: str) -> int:
        fl = (flower or "").title()
        return prio.index(fl) if fl in prio else 9999
    candidates.sort(key=lambda x: (prio_index(x.get("flower")), x.get("price_inr", 999999)))

    # 3) fallback if none
    if not candidates:
        base = sorted(CATALOG, key=lambda x: x.get("price_inr", 999999))
        seen_imgs, chosen = set(), []
        for c in base:
            img = (c.get("image_url") or c.get("image",""))
            if img not in seen_imgs:
                seen_imgs.add(img)
                chosen.append(c)
            if len(chosen) == 3:
                break
        return [shape_item(x) for x in chosen]

    # 4) compute intended price buckets (low/mid/high)
    prices = sorted(sorted({c["price_inr"] for c in candidates}))
    if len(prices) >= 3:
        targets = [prices[0], prices[len(prices)//2], prices[-1]]
    elif len(prices) == 2:
        targets = [prices[0], prices[0], prices[1]]
    else:
        targets = [prices[0], prices[0], prices[0]]

    # 5) pick one per bucket, ensuring distinct images
    chosen, seen_imgs = [], set()

    def try_pick(price):
        for c in candidates:
            if c["price_inr"] == price:
                img = (c.get("image_url") or c.get("image",""))
                if img not in seen_imgs:
                    seen_imgs.add(img)
                    return c
        return None

    for p in targets:
        item = try_pick(p)
        if item and item not in chosen:
            chosen.append(item)

    # 6) pad with next best that has a new image
    for c in candidates:
        if len(chosen) >= 3:
            break
        img = (c.get("image_url") or c.get("image",""))
        if img not in seen_imgs and c not in chosen:
            seen_imgs.add(img)
            chosen.append(c)

    # 7) final safety pad (if catalog is very small)
    i = 0
    while len(chosen) < 3 and i < len(candidates):
        if candidates[i] not in chosen:
            chosen.append(candidates[i])
        i += 1

    return [shape_item(x) for x in chosen[:3]]

def refine_copy(items: List[Dict[str, Any]], category: str) -> List[Dict[str, Any]]:
    """Lightly adjust title/desc by price band so the 3 cards don't read the same."""
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
        if p == low and " • " not in it["title"]:
            it["title"] = f"{it['title']} • Classic"
            it["desc"] = low_t
        elif p == mid and " • " not in it["title"]:
            it["title"] = f"{it['title']} • Signature"
            it["desc"] = mid_t
        elif p == high and " • " not in it["title"]:
            it["title"] = f"{it['title']} • Luxury"
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
        items = refine_copy(items, category)  # subtle copy variation

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
