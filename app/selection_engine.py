# app/selection_engine.py  — Strict Tier Scaffold (2 MIX + 1 MONO, tier-ordered)
# Ultra-beginner-safe; deterministic; rules-first; palette-aware adjacent fallback
# Always returns exactly 3 items (Classic, Signature, Luxury — one per tier).
# MONO is a format (not a tier) and “floats” to its tier position.
# Pricing floors + LG (Luxury-Grand) policy enforced from rules/*.json.
# Editorial redirection notes preserved on MIX items. Strong-intent overrides supported.
from __future__ import annotations

import json, os, re, hashlib
from typing import Any, Dict, List, Optional

# -------------------------------
# File helpers
# -------------------------------
ROOT = os.path.dirname(__file__)
def _p(*parts: str) -> str:
    return os.path.join(ROOT, *parts)

def _load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default

# -------------------------------
# Load rule tables & catalog
# -------------------------------
EMO  = _load_json(_p("rules","emotion_keywords.json"), {"anchors": [], "exact": {}, "keywords": {}})
SUBS = _load_json(_p("rules","substitution_map.json"), {})
WEI  = _load_json(_p("rules","weights_default.json"), {"flower_weights":{}, "emotion_bias":{}})
PRICE= _load_json(_p("rules","pricing_policy.json"), {"currency":"INR","floors":{},"luxury_grand_floor":4999})
TIERP= _load_json(_p("rules","tier_policy.json"), {"luxury_grand":{"allowed_emotions":[],"blocked_emotions":[],"max_per_triad":1,"allow_in_mix":True,"allow_in_mono":True}})
CATALOG: List[Dict[str,Any]] = _load_json(_p("catalog.json"), [])

# -------------------------------
# Constants & guides
# -------------------------------
ICONIC = {"rose","lily","orchid"}
CURRENCY = PRICE.get("currency","INR")

# Emotion adjacency for MIX borrowing
ADJACENT_EMOTIONS: Dict[str, List[str]] = {
    "Romance":       ["Gratitude", "Birthday", "Friendship"],
    "Sympathy":      ["GetWell", "Encouragement"],
    "Celebration":   ["Birthday", "Gratitude"],
    "Encouragement": ["Friendship", "GetWell"],
    "Gratitude":     ["Romance", "Friendship"],
    "Friendship":    ["Encouragement", "Birthday"],
    "GetWell":       ["Sympathy", "Encouragement"],
    "Birthday":      ["Celebration", "Friendship"]
}

# Palette anchors by emotion (loose tokens; MIX fallback prefers a palette match)
EMOTION_PALETTE_GUIDE: Dict[str, List[str]] = {
    "Romance":      ["blush","soft-pink","red","rose","cream","ivory"],
    "Sympathy":     ["white","ivory","cream","soft-green","pastel"],
    "Celebration":  ["bright","yellow","orange","fuchsia","vibrant","multicolor"],
    "Encouragement":["warm","yellow","orange","peach","sunny"],
    "Gratitude":    ["soft-pink","peach","cream","pastel"],
    "Friendship":   ["yellow","sunny","bright","cheerful"],
    "GetWell":      ["white","green","soft","pastel","peach"],
    "Birthday":     ["bright","pink","purple","yellow","fun"]
}

# Tier ordering (final sort)
TIER_RANK = {"Classic": 0, "Signature": 1, "Luxury": 2}
TIERS = ["Classic", "Signature", "Luxury"]

def order_by_tier(items: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    def key(it):
        return (TIER_RANK.get(it.get("tier"), 99), 1 if it.get("luxury_grand") else 0)
    return sorted(items, key=key)

# -------------------------------
# Text & hashing utils
# -------------------------------
def normalize(text: str) -> str:
    if not text: return ""
    t = text.strip().lower()
    t = re.sub(r"https?://\S+","", t)
    t = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+","", t)
    t = re.sub(r"\s+"," ", t)
    return t

def _hash01(seed: str) -> float:
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    n = int(h[:8], 16) % 10000
    return n / 10000.0

# -------------------------------
# Emotion detection (exact > keyword > context > default)
# -------------------------------
def detect_emotion(norm: str, context: Optional[Dict[str,Any]]) -> str:
    for phrase, emo in (EMO.get("exact") or {}).items():
        if phrase in norm: return emo
    counts = {a:0 for a in (EMO.get("anchors") or [])}
    for kw, emo in (EMO.get("keywords") or {}).items():
        if kw in norm: counts[emo] = counts.get(emo,0) + 1
    if counts:
        best = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[0]
        if best[1] > 0: return best[0]
    if context and context.get("emotion_hint") in (EMO.get("anchors") or []):
        return context["emotion_hint"]
    return "Encouragement"  # beginner-safe default

# -------------------------------
# Weights & deterministic pick
# -------------------------------
def weight_for_item(item: Dict[str,Any], emotion: str) -> int:
    flowers = item.get("flowers") or []
    primary = (flowers[0] if flowers else "rose").lower()
    base = int(WEI.get("flower_weights",{}).get(primary, 50))
    bias = int(WEI.get("emotion_bias",{}).get(emotion,{}).get(primary, base))
    return max(1, round((base + bias)/2))

def pick_deterministic(pool: List[Dict[str,Any]], k: int, seed: str, emotion: str) -> List[Dict[str,Any]]:
    scored = []
    for it in pool:
        w = weight_for_item(it, emotion)
        rnd = _hash01(f"{seed}::{it['id']}")  # stable per SKU+seed
        score = rnd / (w/100.0)               # higher weight -> lower score -> earlier
        scored.append((score, it))
    scored.sort(key=lambda x: x[0])
    out, seen = [], set()
    for _, it in scored:
        if len(out) >= k: break
        if it["id"] in seen: continue
        out.append(it); seen.add(it["id"])
    return out

# -------------------------------
# Policy checks
# -------------------------------
def passes_pricing_floor(item: Dict[str,Any]) -> bool:
    floors = PRICE.get("floors", {})
    lg_floor = int(PRICE.get("luxury_grand_floor", 4999))
    tier = item.get("tier","Classic")
    if item.get("luxury_grand"):
        return int(item.get("price_inr", 0)) >= lg_floor
    return int(item.get("price_inr", 0)) >= int(floors.get(tier, 0))

def lg_allowed_for_emotion(item: Dict[str,Any], emotion: str) -> bool:
    if not item.get("luxury_grand"): return True
    lg = TIERP.get("luxury_grand", {})
    if emotion in (lg.get("blocked_emotions") or []): return False
    allowed = lg.get("allowed_emotions") or []
    return (emotion in allowed) if allowed else True

def _has_palette(item: Dict[str,Any]) -> bool:
    pal = item.get("palette")
    return isinstance(pal, list) and len(pal) > 0

def _palette_match(tokens: List[str], desired: List[str]) -> bool:
    if not tokens or not desired: return False
    return bool(set(t.lower() for t in tokens) & set(d.lower() for d in desired))

# -------------------------------
# Context boosts (budget & packaging)
# -------------------------------
def apply_context_preferences(pool: List[Dict[str,Any]], context: Optional[Dict[str,Any]]) -> List[Dict[str,Any]]:
    if not context: return pool[:]
    budget = context.get("budget_inr")
    pref_pack = (context.get("packaging_pref") or "").lower()
    def score(x):
        s = 0
        if pref_pack and (x.get("packaging","").lower()==pref_pack): s += 10
        if isinstance(budget,(int,float)):
            s += 6 if x.get("price_inr",10**9) <= budget else -2
        return s
    return sorted(pool, key=score, reverse=True)

# -------------------------------
# Intent & redirection
# -------------------------------
def strong_intent_override(norm: str) -> Dict[str,Any]:
    out = {"forceIconic": None, "forbidMono": None, "note": None}
    m = re.search(r"\bonly\s+(roses?|lilies|orchids?)\b", norm)
    if m:
        w = m.group(1)
        if w.startswith("rose"): out["forceIconic"] = "rose"
        elif w.startswith("lil"): out["forceIconic"] = "lily"
        elif w.startswith("orchid"): out["forceIconic"] = "orchid"
    m2 = re.search(r"\bmono\s+([a-z]+)\b", norm)
    if m2:
        flower = m2.group(1)
        if flower not in ICONIC:
            out["forbidMono"] = flower
            out["note"] = f"Requested mono {flower} is unavailable; offering nearest alternatives."
    return out

def editorial_redirection(norm: str) -> Optional[Dict[str,Any]]:
    for k, subs in (SUBS or {}).items():
        if k in norm:
            return {"redirectedFrom": k, "substitutes": subs,
                    "note": f"“{k}” is unavailable/seasonal. Redirected to: {', '.join(subs)}."}
    return None

# -------------------------------
# Filtering helpers
# -------------------------------
def _base_ok(x: Dict[str,Any], emotion: str) -> bool:
    return passes_pricing_floor(x) and lg_allowed_for_emotion(x, emotion) and _has_palette(x)

def _global_iconic_mono(emotion: str) -> List[Dict[str,Any]]:
    return [x for x in CATALOG if x.get("mono") and any((f or '').lower() in ICONIC for f in (x.get("flowers") or [])) and _base_ok(x, emotion)]

# Tier-targeted MIX fallbacks
def _adjacent_mix_for_tier(emotion: str, tier: str) -> List[Dict[str,Any]]:
    desired = EMOTION_PALETTE_GUIDE.get(emotion, [])
    out: List[Dict[str,Any]] = []
    for adj in ADJACENT_EMOTIONS.get(emotion, []):
        for x in CATALOG:
            if x.get("mono"): 
                continue
            if x.get("tier") != tier:
                continue
            if x.get("emotion") != adj:
                continue
            if not _base_ok(x, adj):
                continue
            if not _palette_match(x.get("palette") or [], desired):
                continue
            out.append(x)
    return out

def _global_mix_for_tier(emotion: str, tier: str) -> List[Dict[str,Any]]:
    return [x for x in CATALOG if (not x.get("mono")) and x.get("tier")==tier and _base_ok(x, emotion)]

# -------------------------------
# Output mapping & validation
# -------------------------------
def _map_out(it: Dict[str,Any], note: Optional[Dict[str,Any]]|str=None) -> Dict[str,Any]:
    note_text = note if isinstance(note, str) else (note.get("note") if isinstance(note, dict) else None)
    out = {
        "id": it["id"],
        "title": it["title"],
        "desc": it["desc"],
        "image": it["image_url"],
        "price": it["price_inr"],
        "currency": CURRENCY,
        "emotion": it["emotion"],
        "tier": it["tier"],
        "packaging": it["packaging"],
        "mono": bool(it.get("mono")),
        "palette": it.get("palette") or [],
        "luxury_grand": bool(it.get("luxury_grand"))
    }
    # Preserve note only on MIX items (per playbook)
    if note_text and not out["mono"]:
        out["note"] = note_text
    return out

def validate_output(items: List[Dict[str,Any]]) -> None:
    if not isinstance(items, list) or len(items)!=3:
        raise ValueError("Output must contain exactly 3 items.")
    if sum(1 for x in items if x["mono"]) != 1:
        raise ValueError("Exactly one MONO item is required.")
    tiers = [it.get("tier") for it in items]
    if set(tiers) != set(["Classic","Signature","Luxury"]):
        raise ValueError(f"Must include one of each tier ['Classic','Signature','Luxury']; got {tiers}.")
    ids=set()
    req = ["id","title","desc","image","price","currency","emotion","tier","packaging","mono","palette","luxury_grand"]
    for it in items:
        for k in req:
            v = it.get(k, None)
            if v is None or (k=="palette" and not v):
                raise ValueError(f"Missing/empty field {k} on {it.get('id','unknown')}")
        if it["id"] in ids: 
            raise ValueError("Duplicate SKU in triad.")
        ids.add(it["id"])

# -------------------------------
# Public API
# -------------------------------
def selection_engine(prompt: str, context: Optional[Dict[str,Any]]=None) -> List[Dict[str,Any]]:
    """Rules-first selection engine.
    Input:
      prompt: user text (1–500 chars)
      context: optional dict with keys: emotion_hint?, budget_inr?, packaging_pref?, locale?
    Returns exactly 3 items (dicts) with required fields; 2 MIX + 1 MONO; tiers are Classic, Signature, Luxury.
    """
    context = context or {}
    norm = normalize(prompt)
    emotion = detect_emotion(norm, context)

    # Base pools for detected emotion
    base = [x for x in CATALOG if x.get("emotion") == emotion and _base_ok(x, emotion)]
    mix_pool  = [x for x in base if not x.get("mono")]
    mono_pool = [x for x in base if x.get("mono") and any((f or '').lower() in ICONIC for f in (x.get("flowers") or []))]

    # Respect LG allow/deny in pools
    lgp = TIERP.get("luxury_grand", {})
    if not lgp.get("allow_in_mix", True):
        mix_pool  = [x for x in mix_pool  if not x.get("luxury_grand")]
    if not lgp.get("allow_in_mono", True):
        mono_pool = [x for x in mono_pool if not x.get("luxury_grand")]

    # Intent & editorial
    intent = strong_intent_override(norm)
    redir  = editorial_redirection(norm)
    note = redir or ({"note": intent.get("note")} if intent.get("note") else None)

    # --- MONO pick (iconic; force if "only roses/lilies/orchids") ---
    def has_species(x, species): 
        return any((species == (f or '').lower()) for f in (x.get("flowers") or []))

    mono_pick = None
    if intent.get("forceIconic"):  # force specific iconic species
        species = intent["forceIconic"]
        pool = [x for x in mono_pool if has_species(x, species)]
        if not pool:
            pool = [x for x in _global_iconic_mono(emotion) if has_species(x, species)]
        if pool:
            pool = apply_context_preferences(pool, context)
            mono_pick = pick_deterministic(pool, 1, seed=f"{emotion}/{norm}/mono_forced", emotion=emotion)[0]

    if not mono_pick:
        pool = mono_pool if mono_pool else _global_iconic_mono(emotion)
        if not pool:
            raise ValueError("No MONO candidate found; add iconic mono SKUs to catalog.")
        pool = apply_context_preferences(pool, context)
        mono_pick = pick_deterministic(pool, 1, seed=f"{emotion}/{norm}/mono", emotion=emotion)[0]

    mono_tier = mono_pick.get("tier")

    # --- MIX picks (strict tier scaffold: fill remaining two tiers) ---
    def pick_mix_for_tier(tier: str) -> Optional[Dict[str,Any]]:
        # 1) same emotion, correct tier
        pool = [x for x in mix_pool if x.get("tier") == tier]
        # 2) adjacent emotion, palette-matched
        if not pool:
            pool = _adjacent_mix_for_tier(emotion, tier)
        # 3) global fallback (any emotion), still policy-safe for detected emotion
        if not pool:
            pool = _global_mix_for_tier(emotion, tier)
        if not pool:
            return None
        pool = apply_context_preferences(pool, context)
        return pick_deterministic(pool, 1, seed=f"{emotion}/{norm}/mix/{tier}", emotion=emotion)[0]

    needed_tiers = [t for t in TIERS if t != mono_tier]
    mix_picks: List[Dict[str,Any]] = []
    for t in needed_tiers:
        cand = pick_mix_for_tier(t)
        if not cand:
            raise ValueError(f"No MIX candidate for tier {t}; add catalog items.")
        mix_picks.append(cand)

    # --- Assemble & order; preserve note on MIX ---
    cards = [_map_out(mix_picks[0], note=note), _map_out(mix_picks[1], note=note), _map_out(mono_pick)]
    cards = order_by_tier(cards)

    # --- Enforce LG cap (≤ max_per_triad) with tier-aware replacement ---
    cap = int(TIERP.get("luxury_grand", {}).get("max_per_triad", 1))
    lg_idxs = [i for i,c in enumerate(cards) if c.get("luxury_grand")]
    if len(lg_idxs) > cap:
        # Prefer replacing the MIX luxury first (keep MONO if possible)
        order = [i for i in lg_idxs if not cards[i]["mono"]] + [i for i in lg_idxs if cards[i]["mono"]]
        for i in order[:len(lg_idxs)-cap]:
            tier = cards[i]["tier"]
            pool = [x for x in _global_mix_for_tier(emotion, tier) if not x.get("luxury_grand")]
            if pool:
                pool = apply_context_preferences(pool, context)
                repl = pick_deterministic(pool, 1, seed=f"{emotion}/{norm}/lg_repl/{tier}", emotion=emotion)[0]
                cards[i] = _map_out(repl, note=note)

    # Final order & validation
    cards = order_by_tier(cards)
    validate_output(cards)
    return cards
