
# app/selection_engine.py
# Ultra-beginner-safe Selection Engine — Phase 1.2 + 1.3 (Edge-Case Playbooks)
# Deterministic, rules-first; strict tier scaffold (Classic → Signature → Luxury).
# Always returns exactly 3 items (2 MIX + 1 MONO) with palette[].
# Edge registers (sympathy/apology/farewell/valentine) apply tone/palette/species rules and copy ≤ N words.
# LG (Luxury-Grand) policy enforced via rules/tier_policy.json, with soft multipliers in registers.

from __future__ import annotations
import json, os, re, hashlib
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------
# File helpers
# ------------------------------
ROOT = os.path.dirname(__file__)
def _p(*parts: str) -> str:
    return os.path.join(ROOT, *parts)

def _load_json(path: str, default: Any) -> Any:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default

# ------------------------------
# Load rule tables & catalog
# ------------------------------
CATALOG: List[Dict[str, Any]] = _load_json(_p('catalog.json'), [])
RULES_DIR = _p('rules')

EMOTION_KEYWORDS = _load_json(os.path.join(RULES_DIR, 'emotion_keywords.json'), {})
SUBSTITUTION_MAP = _load_json(os.path.join(RULES_DIR, 'substitution_map.json'), {})
WEIGHTS_DEFAULT = _load_json(os.path.join(RULES_DIR, 'weights_default.json'), {})
PRICING_POLICY = _load_json(os.path.join(RULES_DIR, 'pricing_policy.json'), {
    "classic_floor_inr": 1499,
    "signature_floor_inr": 2499,
    "luxury_floor_inr": 3999,
    "luxury_grand_floor_inr": 4999
})
TIER_POLICY = _load_json(os.path.join(RULES_DIR, 'tier_policy.json'), {
    "luxury_grand": {
        "allowed_emotions": [],
        "blocked_emotions": ["Sympathy","GetWell","Encouragement"],
        "max_per_triad": 1,
        "allow_in_mix": True,
        "allow_in_mono": True
    }
})
EDGE_KEYWORDS = _load_json(os.path.join(RULES_DIR, 'edge_keywords.json'), {
    "sympathy": ["sorry for your loss","condolence","grief","passing"],
    "apology": ["sorry","apologize","forgive"],
    "farewell": ["goodbye","farewell","parting","last day"],
    "valentine": ["valentine","love","anniversary","romantic"],
    "grand_intent_keywords": ["grand","bigger","large","lavish","extravagant","50+","hundred","massive","most beautiful"],
    "relationship_grandeur_cues": ["boss","manager","director","ceo","founder","mentor","teacher","professor","parents","grandparents","board"]
})
EDGE_REGISTERS = _load_json(os.path.join(RULES_DIR, 'edge_registers.json'), {
    "copy_max_words": 20,
    "sympathy": {
        "edge_case": True,
        "tone": "muted",
        "palette_targets": ["white","cream","soft-green","ivory"],
        "species_prefer": ["lily","orchid","chrysanthemum"],
        "species_avoid": ["sunflower","gerbera","neon-mixed"],
        "lg_policy": "block",
        "lg_weight_multiplier": 0.0,
        "min_budget_for_lg": 999999
    },
    "apology": {
        "edge_case": True,
        "tone": "gentle",
        "palette_targets": ["white","soft-purple","lavender","blush","soft-green"],
        "species_prefer": ["orchid","lily","rose"],
        "species_avoid": [],
        "lg_policy": "allow_soft",
        "lg_weight_multiplier": 0.8,
        "min_budget_for_lg": 4999
    },
    "farewell": {
        "edge_case": True,
        "tone": "respectful",
        "palette_targets": ["white","cream","peach","soft-green"],
        "species_prefer": ["rose","lily","orchid"],
        "species_avoid": ["neon-mixed"],
        "lg_policy": "allow_soft",
        "lg_weight_multiplier": 0.8,
        "min_budget_for_lg": 4999
    },
    "valentine": {
        "edge_case": True,
        "tone": "bold",
        "palette_targets": ["red","blush","crimson"],
        "species_prefer": ["rose"],
        "species_avoid": [],
        "lg_policy": "allow_strong",
        "lg_weight_multiplier": 1.5,
        "min_budget_for_lg": 3999
    }
})

ICONIC = {"rose","lily","orchid"}
TIER_RANK = {"Classic": 0, "Signature": 1, "Luxury": 2}

# ------------------------------
# Normalization & detection
# ------------------------------
_URL_RE = re.compile(r'https?://\S+|www\.\S+', re.I)
_EMAIL_RE = re.compile(r'\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b')

def normalize(prompt: str) -> str:
    s = (prompt or "").strip()
    s = _URL_RE.sub('', s)
    s = _EMAIL_RE.sub('', s)
    s = re.sub(r'\s+', ' ', s)
    return s.lower()

def _kw_hit(text: str, words: List[str]) -> bool:
    t = text.lower()
    return any(w in t for w in words)

def detect_edge_case(prompt: str) -> Optional[str]:
    p = prompt.lower()
    for k in ("sympathy","apology","farewell","valentine"):
        if _kw_hit(p, EDGE_KEYWORDS.get(k, [])):
            return k
    return None

def detect_emotion(prompt: str, context: Optional[Dict[str,Any]]=None) -> str:
    p = prompt.lower()
    # exact route first
    for emo, keys in EMOTION_KEYWORDS.items():
        if any(p == k for k in keys):
            return emo
    # keyword route
    for emo, keys in EMOTION_KEYWORDS.items():
        if _kw_hit(p, keys):
            return emo
    # context hint
    if context and isinstance(context.get("emotion_hint"), str):
        return context["emotion_hint"]
    # default
    return "Romance"

# ------------------------------
# Policy helpers
# ------------------------------
def _meets_floor(item: Dict[str,Any]) -> bool:
    tier = item.get("tier")
    price = int(item.get("price_inr") or item.get("price") or 0)
    if tier == "Classic":
        return price >= int(PRICING_POLICY.get("classic_floor_inr", 1499))
    if tier == "Signature":
        return price >= int(PRICING_POLICY.get("signature_floor_inr", 2499))
    if tier == "Luxury":
        if item.get("luxury_grand"):
            return price >= int(PRICING_POLICY.get("luxury_grand_floor_inr", 4999))
        return price >= int(PRICING_POLICY.get("luxury_floor_inr", 3999))
    return True

def _lg_allowed(emotion: str, item: Dict[str,Any]) -> bool:
    if not item.get("luxury_grand"):
        return True
    lg = TIER_POLICY.get("luxury_grand", {})
    blocked = set(lg.get("blocked_emotions", []))
    allowed = set(lg.get("allowed_emotions", []))
    if emotion in blocked:
        return False
    if allowed:  # allow-list present
        return emotion in allowed
    return True  # allow all others if allow-list empty

# ------------------------------
# Scoring helpers
# ------------------------------
def _species_set(item: Dict[str,Any]) -> set:
    return set(map(str.lower, item.get("flowers") or []))

def _palette_set(item: Dict[str,Any]) -> set:
    return set(map(str.lower, item.get("palette") or []))

def _packaging_match_boost(item: Dict[str,Any], context: Optional[Dict[str,Any]]) -> float:
    if not context: return 1.0
    pref = context.get("packaging_pref")
    if pref and str(item.get("packaging","")).lower() == str(pref).lower():
        return 1.08
    return 1.0

def _budget_boost(item: Dict[str,Any], context: Optional[Dict[str,Any]]) -> float:
    if not context: return 1.0
    try:
        b = float(context.get("budget_inr") or 0)
    except Exception:
        return 1.0
    if b <= 0: return 1.0
    price = float(item.get("price_inr") or item.get("price") or 0)
    # soft scoring: slight boost if within budget, slight penalty if over
    if price <= b:
        return 1.10
    if price <= b * 1.2:
        return 1.02
    return 0.95

def _edge_lg_multiplier(edge: Optional[str], context: Optional[Dict[str,Any]]) -> float:
    if not edge: return 1.0
    reg = EDGE_REGISTERS.get(edge, {})
    pol = reg.get("lg_policy", "allow_soft")
    if pol == "block":
        return 0.0
    mult = float(reg.get("lg_weight_multiplier", 1.0))
    min_b = float(reg.get("min_budget_for_lg", 0))
    budget = 0.0
    try:
        if context and context.get("budget_inr"):
            budget = float(context["budget_inr"])
    except Exception:
        budget = 0.0
    if budget >= min_b:
        return mult
    # grand intent override
    p = (context or {}).get("_normalized_prompt","")
    if p and any(k in p for k in EDGE_KEYWORDS.get("grand_intent_keywords", [])):
        return max(1.0, mult)  # lift to neutral-or-strong
    return 0.8 if pol == "allow_soft" else mult

def _edge_weight_adjust(item: Dict[str,Any], edge: Optional[str]) -> float:
    if not edge: return 1.0
    reg = EDGE_REGISTERS.get(edge, {})
    w = 1.0
    if reg:
        sp = _species_set(item)
        pal = _palette_set(item)
        prefer = set(map(str.lower, reg.get("species_prefer", [])))
        avoid = set(map(str.lower, reg.get("species_avoid", [])))
        targets = set(map(str.lower, reg.get("palette_targets", [])))
        if prefer and sp.intersection(prefer):
            w *= 1.35
        if avoid and sp.intersection(avoid):
            w *= 0.75
        if targets and pal.intersection(targets):
            w *= 1.15
    return w

def _score_item(item: Dict[str,Any], context: Optional[Dict[str,Any]], edge: Optional[str]) -> float:
    base = float(item.get("weight", 50))
    # default weight by species fallback
    for f in item.get("flowers") or []:
        species = str(f).lower()
        base *= float(WEIGHTS_DEFAULT.get(species, 1.0))
    # gentle boosts
    base *= _packaging_match_boost(item, context)
    base *= _budget_boost(item, context)
    base *= _edge_weight_adjust(item, edge)
    # LG multiplier
    if item.get("luxury_grand"):
        base *= _edge_lg_multiplier(edge, context)
    return base

def _truncate_words(text: str, max_words: int) -> str:
    if max_words <= 0: return ""
    words = re.findall(r'\S+', text or "")
    if len(words) <= max_words:
        return text or ""
    return " ".join(words[:max_words]).strip()

def order_by_tier(items: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    def key(it):
        rank = TIER_RANK.get(it.get("tier"), 99)
        lg_bias = 1 if it.get("luxury_grand") else 0
        return (rank, lg_bias)
    return sorted(items, key=key)

# ------------------------------
# Engine
# ------------------------------
def selection_engine(prompt: str, context: Optional[Dict[str,Any]]=None) -> List[Dict[str,Any]]:
    if context is None:
        context = {}
    p_norm = normalize(prompt)
    context["_normalized_prompt"] = p_norm  # for grand intent checks

    # Detect routes
    edge = detect_edge_case(p_norm)
    emotion = detect_emotion(p_norm, context)

    # Base pool gated by emotion + policy floors
    base = [x for x in CATALOG if str(x.get("emotion")).lower() == str(emotion).lower()]
    base = [x for x in base if _meets_floor(x) and _lg_allowed(emotion, x)]

    # Strong-intent override for iconic MONO
    force_mono_species: Optional[str] = None
    for sp in ICONIC:
        if re.search(rf"\bonly\s+{sp}s?\b", p_norm):
            force_mono_species = sp
            break

    # Editorial redirection — if prompt requests unavailable/seasonal flower
    redirect_note: Optional[str] = None
    for requested, subs in SUBSTITUTION_MAP.items():
        if requested.lower() in p_norm and requested.lower() not in (f.lower() for x in base for f in (x.get("flowers") or [])):
            redirect_note = f"Nearest alternative to your unavailable request: {requested}."
            break

    # Pools
    iconic_mono = []
    mix = []
    for x in base:
        if x.get("mono"):
            if ICONIC.intersection(_species_set(x)):
                iconic_mono.append(x)
        else:
            mix.append(x)

    # If mono pool empty, borrow iconic across catalog
    if not iconic_mono:
        for x in CATALOG:
            if x.get("mono") and ICONIC.intersection(_species_set(x)) and _meets_floor(x) and _lg_allowed(emotion, x):
                iconic_mono.append(x)

    # Scoring
    def scored(arr: List[Dict[str,Any]]) -> List[Tuple[float, Dict[str,Any]]]:
        return sorted((( _score_item(it, context, edge), it) for it in arr), key=lambda t: t[0], reverse=True)

    mix_scored = scored(mix)
    mono_scored = scored(iconic_mono)

    # Best per tier (mix + mono)
    tiers = ["Classic","Signature","Luxury"]
    best_mix_by_tier: Dict[str, Optional[Dict[str,Any]]] = {t: None for t in tiers}
    best_mono_by_tier: Dict[str, Optional[Dict[str,Any]]] = {t: None for t in tiers}
    for score, it in mix_scored:
        t = it.get("tier")
        if t in tiers and best_mix_by_tier[t] is None:
            best_mix_by_tier[t] = it
    for score, it in mono_scored:
        t = it.get("tier")
        if t in tiers and best_mono_by_tier[t] is None:
            best_mono_by_tier[t] = it

    # Decide tier for MONO
    mono_tier: Optional[str] = None
    if force_mono_species:
        for t in tiers:
            cand = best_mono_by_tier.get(t)
            if cand and force_mono_species in _species_set(cand):
                mono_tier = t
                break
    if mono_tier is None:
        for t in tiers:
            if best_mix_by_tier.get(t) is None and best_mono_by_tier.get(t) is not None:
                mono_tier = t
                break
    if mono_tier is None:
        deltas = []
        for t in tiers:
            m = best_mix_by_tier.get(t)
            mo = best_mono_by_tier.get(t)
            if mo is None:
                continue
            score_m = next((s for s,i in mix_scored if i is m), 0.0)
            score_mo = next((s for s,i in mono_scored if i is mo), 0.0)
            deltas.append((score_mo - score_m, t))
        if deltas:
            deltas.sort(reverse=True)
            mono_tier = deltas[0][1]

    # Assemble one-per-tier
    out: List[Dict[str,Any]] = []
    used_ids = set()
    for t in tiers:
        it = None
        if t == mono_tier and best_mono_by_tier.get(t):
            it = best_mono_by_tier[t]
        elif best_mix_by_tier.get(t):
            it = best_mix_by_tier[t]

        if not it:
            # Palette-aware tier fallback (emotion-safe)
            targets = set()
            if edge:
                targets = set(map(str.lower, EDGE_REGISTERS.get(edge, {}).get("palette_targets", [])))

            def good(x):
                return (
                    x.get('tier') == t
                    and x.get('id') not in used_ids
                    and _meets_floor(x)
                    and x in base  # keep emotion gate
                )
            any_tier_item = next(
                (x for x in base
                 if good(x)
                 and targets
                 and targets.intersection(_palette_set(x))),
                None
            )
            if not any_tier_item:
                any_tier_item = next((x for x in base if good(x)), None)
            if any_tier_item:
                it = any_tier_item

        if it:
            used_ids.add(it.get("id"))
            out.append(it)

    # Last-resort backfill if thin
    if len(out) < 3:
        for t in tiers:
            if len(out) >= 3: break
            if not any(x.get("tier") == t for x in out):
                any_tier = next((x for x in CATALOG if x.get("tier")==t and x.get("id") not in used_ids and _meets_floor(x)), None)
                if any_tier:
                    out.append(any_tier)
                    used_ids.add(any_tier.get("id"))

    # Map to output schema
    copy_cap = int(EDGE_REGISTERS.get("copy_max_words", 20)) if edge else 9999
    triad = []
    for it in out[:3]:
        row = {
            "id": it.get("id"),
            "title": it.get("title"),
            "desc": _truncate_words(it.get("desc",""), copy_cap) if edge else (it.get("desc") or ""),
            "image": it.get("image_url") or it.get("image"),
            "price": int(it.get("price", it.get("price_inr", 0)) or 0),
            "currency": "INR",
            "emotion": it.get("emotion"),
            "tier": it.get("tier"),
            "packaging": it.get("packaging"),
            "mono": bool(it.get("mono")),
            "palette": it.get("palette") or [],
            "luxury_grand": bool(it.get("luxury_grand")),
        }
        if redirect_note and not row["mono"]:
            row["note"] = redirect_note
        if edge:
            row["edge_case"] = True
            row["edge_register"] = edge
        triad.append(row)

    # Ensure exactly one MONO
    mono_count = sum(1 for x in triad if x["mono"])
    if mono_count == 0 and mono_scored:
        used = {x["id"] for x in triad}
        repl_tier = None
        for _, mo in mono_scored:
            if mo.get("id") in used: 
                continue
            repl_tier = mo.get("tier")
            break
        if repl_tier:
            for i, x in enumerate(triad):
                if x["tier"] == repl_tier and not x["mono"]:
                    mo = next((mo for _,mo in mono_scored if mo.get("tier")==repl_tier and mo.get("id") not in used), None)
                    if mo:
                        triad[i] = {
                            "id": mo.get("id"),
                            "title": mo.get("title"),
                            "desc": _truncate_words(mo.get("desc",""), copy_cap) if edge else (mo.get("desc") or ""),
                            "image": mo.get("image_url") or mo.get("image"),
                            "price": int(mo.get("price", mo.get("price_inr", 0)) or 0),
                            "currency": "INR",
                            "emotion": mo.get("emotion"),
                            "tier": mo.get("tier"),
                            "packaging": mo.get("packaging"),
                            "mono": True,
                            "palette": mo.get("palette") or [],
                            "luxury_grand": bool(mo.get("luxury_grand")),
                        }
                        if edge:
                            triad[i]["edge_case"] = True
                            triad[i]["edge_register"] = edge
                        break
    elif mono_count > 1:
        used = {x["id"] for x in triad}
        mix_candidates = [it for _, it in mix_scored if it.get("id") not in used and _meets_floor(it)]
        i = 0
        while mono_count > 1 and i < len(triad) and mix_candidates:
            if triad[i]["mono"]:
                replacement = mix_candidates.pop(0)
                triad[i] = {
                    "id": replacement.get("id"),
                    "title": replacement.get("title"),
                    "desc": _truncate_words(replacement.get("desc",""), copy_cap) if edge else (replacement.get("desc") or ""),
                    "image": replacement.get("image_url") or replacement.get("image"),
                    "price": int(replacement.get("price", replacement.get("price_inr", 0)) or 0),
                    "currency": "INR",
                    "emotion": replacement.get("emotion"),
                    "tier": replacement.get("tier"),
                    "packaging": replacement.get("packaging"),
                    "mono": False,
                    "palette": replacement.get("palette") or [],
                    "luxury_grand": bool(replacement.get("luxury_grand")),
                }
                if edge:
                    triad[i]["edge_case"] = True
                    triad[i]["edge_register"] = edge
                mono_count -= 1
            i += 1

    # Final deterministic tier order
    triad = order_by_tier(triad)[:3]

    return triad

# Convenience for local debugging
if __name__ == "__main__":
    tests = [
        ("romantic anniversary under 2000", {"budget_inr": 2000}),
        ("only lilies please", {}),
        ("hydrangea bouquet", {}),
        ("i'm so sorry for your loss", {}),
        ("valentine surprise", {"budget_inr": 6000})
    ]
    for t, ctx in tests:
        out = selection_engine(t, ctx)
        print("\n=== ", t)
        for o in out:
            print(o["tier"], "MIX" if not o["mono"] else "MONO", "LG" if o.get("luxury_grand") else "", "-", o["title"])
