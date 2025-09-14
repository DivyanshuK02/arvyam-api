# app/selection_engine.py
# Ultra-beginner-safe Selection Engine — Phase 1.2 + 1.3 (Edge-Case Playbooks)
# - Deterministic, rules-first; strict tier scaffold (Classic → Signature → Luxury).
# - Always returns exactly 3 items (2 MIX + 1 MONO) with palette[].
# - Edge registers (sympathy/apology/farewell/valentine): tone/palette/species rules and copy ≤ N words.
# - LG policy: emotion block-list from rules/tier_policy.json and soft multipliers in registers (intent-only; no numeric budgets in code).
# - External contract: export curate(), selection_engine(), normalize(), detect_emotion().
#
# Notes
# -----
# * No network calls. Reads JSON files from /app and /app/rules only.
# * This module avoids hard-coded prices or numeric budgets (per project direction).

from __future__ import annotations

import json, os, re, hashlib
from typing import Any, Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException, Query # Added Query import

__all__ = ["curate", "selection_engine", "normalize", "detect_emotion"]

# ------------------------------------------------------------
# File helpers
# ------------------------------------------------------------

ROOT = os.path.dirname(__file__)

def _p(*parts: str) -> str:
    return os.path.join(ROOT, *parts)

def _load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default

# ------------------------------------------------------------
# Data sources (all project-owned JSON files)
# ------------------------------------------------------------

CATALOG = _load_json(_p("catalog.json"), [])
RULES_DIR = _p("rules")
EMOTION_KEYWORDS = _load_json(os.path.join(RULES_DIR, "emotion_keywords.json"), {})
EDGE_KEYWORDS = _load_json(os.path.join(RULES_DIR, "edge_keywords.json"), {})
EDGE_REGISTERS = _load_json(os.path.join(RULES_DIR, "edge_registers.json"), {})
TIER_POLICY = _load_json(os.path.join(RULES_DIR, "tier_policy.json"), {"luxury_grand": {"blocked_emotions": []}})
SUB_NOTES = _load_json(os.path.join(RULES_DIR, "substitution_notes.json"), {"default": "Requested {from} is seasonal/unavailable; offering {alt} as the nearest alternative."})
# Edge register keys (strict): only these four are considered "edge cases".
EDGE_CASE_KEYS = {"sympathy", "apology", "farewell", "valentine"}

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

TIER_ORDER = ["Classic", "Signature", "Luxury"]
TIER_RANK = {t: i for i, t in enumerate(TIER_ORDER)}

ICONIC_SPECIES = {
    "lily": ["lily", "lilies"],
    "rose": ["rose", "roses"],
    "orchid": ["orchid", "orchids"]
}

ONLY_ICONIC_RE = re.compile(r"\bonly\s+(lil(?:y|ies)|roses?|orchids?)\b", re.I)

VAGUE_TERMS = {"nice", "beautiful", "pretty", "some", "any", "simple", "best", "good", "flowers", "flower"}
GRAND_INTENT = {"grand","bigger","large","lavish","extravagant","50+","hundred","massive","most beautiful"}

def normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s[:500]

def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z]+", normalize(s))

def _any_match(text: str, phrases: List[str]) -> bool:
    t = normalize(text)
    for p in phrases:
        if p and normalize(p) in t:
            return True
    return False

def _intent_clarity(prompt: str, matched_keywords: int) -> float:
    """Return 0.0 when prompt is very generic/unclear; else 1.0."""
    p = normalize(prompt)
    words = re.findall(r"[a-z]+", p)
    if matched_keywords == 0 and (len(words) <= 4 or all(w in VAGUE_TERMS for w in words)):
        return 0.0
    return 1.0

def _has_grand_intent(prompt: str) -> bool:
    p = normalize(prompt)
    return any(w in p for w in GRAND_INTENT)

def _add_unclear_mix_note(triad: list) -> None:
    """Attach a single gentle clarifying note to the first MIX card, if none already."""
    for it in triad:
        if not it.get("mono") and not it.get("note"):
            it["note"] = "Versatile picks while you decide — tell us the occasion for a more personal curation."
            break

def _contains_any(text: str, terms: list[str]) -> bool:
    t = text.lower()
    return any((term or "").lower() in t for term in (terms or []))

def _matches_regex_list(text: str, patterns: list[str]) -> bool:
    if not patterns: return False
    for pat in patterns:
        try:
            if re.search(pat, text, flags=re.I):
                return True
        except re.error:
            continue
    return False

def _has_proximity(text: str, a: str, b: str, window: int) -> bool:
    if not a or not b or window <= 0: return False
    tokens = re.findall(r"\w+", text.lower())
    pos_a = [i for i,w in enumerate(tokens) if w == a.lower()]
    pos_b = [i for i,w in enumerate(tokens) if w == b.lower()]
    return any(abs(i-j) <= window for i in pos_a for j in pos_b)

def _is_false_friend(text: str, phrases: list[str]) -> bool:
    return _contains_any(text, phrases or [])

def detect_emotion(prompt: str, context: dict | None) -> str:
    p = (prompt or "").strip().lower()

    # 0) Safe context hint (only accept known anchors)
    hint = (context or {}).get("emotion_hint")
    anchors = EMOTION_KEYWORDS.get("anchors", [])
    if isinstance(hint, str) and hint in anchors:
        return hint

    # Tables
    exact_map = EMOTION_KEYWORDS.get("exact", {}) or {}
    combos    = EMOTION_KEYWORDS.get("combos", []) or []
    buckets   = EMOTION_KEYWORDS.get("keywords", {}) or {}
    disamb    = EMOTION_KEYWORDS.get("disambiguation", []) or []

    # 1) Disambiguation rules (highest precision)
    for rule in disamb:
        when_any = rule.get("when_any", [])
        if not when_any or not any(w in p for w in map(str.lower, when_any)):
            continue
        for branch in rule.get("route", []):
            if_any = branch.get("if_any", [])
            if if_any and any(t in p for t in map(str.lower, if_any)):
                return branch.get("anchor")
            if "else" in branch:
                return branch["else"]

    # 2) Edge keywords are handled elsewhere; here we do emotion rails.

    # 3) Exact phrase map (case-insensitive contains)
    for phrase, anchor in exact_map.items():
        if phrase.lower() in p:
            return anchor

    # 4) Combos: all terms present
    for combo in combos:
        terms = [t.lower() for t in combo.get("all", [])]
        if terms and all(t in p for t in terms):
            return combo.get("anchor")

    # 5) Enriched (optional) — regex & proximity_pairs
    enr = EMOTION_KEYWORDS.get("enriched", {}) or {}
    if enr.get("false_friends") and _is_false_friend(p, enr["false_friends"]):
        # fall through to buckets without enriched scoring
        pass
    else:
        # regex with anchor
        for spec in enr.get("regex", []):
            if _matches_regex_list(p, [spec.get("pattern","")]):
                return spec.get("anchor")
        # proximity pairs with anchor
        for spec in enr.get("proximity_pairs", []):
            if _has_proximity(p, spec.get("a",""), spec.get("b",""), int(spec.get("window",2))):
                return spec.get("anchor")

    # 6) Buckets: score by keyword hits; tie-break by anchors[] order
    scores = {a: 0 for a in anchors}
    for a, words in (buckets or {}).items():
        scores[a] = sum(1 for w in (words or []) if w.lower() in p)

    # pick best; if tie, prefer earlier in anchors[]
    best = max(anchors, key=lambda a: (scores.get(a,0), -anchors.index(a))) if anchors else None
    if best and scores.get(best, 0) > 0:
        return best

    # 7) Fallback default
    return anchors[0] if anchors else "Affection/Support"

EDGE_CASE_KEYS = {"sympathy","apology","farewell","valentine"}

def detect_edge_register(prompt: str) -> str | None:
    p = (prompt or "").strip().lower()
    if not p: return None

    # 1) Priority order (highest first)
    order = ["sympathy","apology","farewell","valentine"]

    for key in order:
        base = EDGE_KEYWORDS.get(key, [])  # backward-compatible arrays
        enr  = EDGE_KEYWORDS.get(f"{key}_enriched", {}) or {}

        # False friends gate
        if _is_false_friend(p, enr.get("false_friends")):
            continue

        # Exact phrases (enriched)
        exact = enr.get("exact", [])
        if exact and _contains_any(p, exact):
            return key

        # “contains_any” (enriched)
        if enr.get("contains_any") and _contains_any(p, enr["contains_any"]):
            return key

        # Regex (enriched)
        if enr.get("regex") and _matches_regex_list(p, enr["regex"]):
            return key

        # Proximity pairs (enriched)
        prox = enr.get("proximity_pairs") or []
        for pair in prox:
            if _has_proximity(p, pair.get("a",""), pair.get("b",""), int(pair.get("window",2))):
                return key

        # Back-compat simple arrays
        if base and _contains_any(p, base):
            return key

    return None

def _has_species(item: Dict[str, Any], species: str) -> bool:
    flowers = [f.lower() for f in item.get("flowers", [])]
    return species.lower() in flowers

def _is_lg(item: Dict[str, Any]) -> bool:
    return bool(item.get("luxury_grand"))

def _is_mono(item: Dict[str, Any]) -> bool:
    return bool(item.get("mono", False))

def _tier_rank(item: Dict[str, Any]) -> int:
    return TIER_RANK.get(item.get("tier"), 99)

def _truncate_words(text: str, max_words: int) -> str:
    words = re.findall(r"\S+", text or "")
    if len(words) <= max_words:
        return text or ""
    return " ".join(words[:max_words]).rstrip(",.;:!—-") + "…"


def _enforce_copy_limit(text: str, edge_type: Optional[str]) -> str:
    root_cap = int(EDGE_REGISTERS.get("copy_max_words", 20))
    reg_cap = int(EDGE_REGISTERS.get(edge_type, {}).get("copy_max_words", root_cap)) if edge_type else root_cap
    cap = max(1, min(reg_cap, 20))  # Phase-1 hard fence
    words = re.findall(r"\S+", text or "")
    if len(words) <= cap:
        return text or ""
    return " ".join(words[:cap]).rstrip(",.;:!—-") + "…"


def _order_by_tier(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(it):
        r = _tier_rank(it)
        # LG should not jump outside Luxury slot; within Luxury, keep LG after non-LG
        lg_bias = 1 if (it.get("tier") == "Luxury" and _is_lg(it)) else 0
        return (r, lg_bias)
    return sorted(items, key=key)

# ------------------------------------------------------------
# Scoring (palette/species + registers, no numeric budgets)
# ------------------------------------------------------------

def _compute_weight(item: Dict[str, Any],
                    base_int: int,
                    register: Dict[str, Any],
                    edge_case: Optional[str],
                    iconic_intent: Optional[str]) -> float:
    w = float(base_int)
    palette = [p.lower() for p in item.get("palette", [])]
    flowers = [f.lower() for f in item.get("flowers", [])]

    # Edge register adjustments
    if edge_case and register:
        ptb = float(register.get("palette_target_boost", 1.0))
        pap = float(register.get("palette_avoid_penalty", 1.0))
        targets = [x.lower() for x in register.get("palette_targets", [])]
        avoids  = [x.lower() for x in register.get("palette_avoid", [])]
        if targets and any(t in palette for t in targets):
            w *= ptb
        if avoids and any(a in palette for a in avoids):
            w *= pap

        spref  = [x.lower() for x in register.get("species_prefer", [])]
        savoid = [x.lower() for x in register.get("species_avoid", [])]
        if spref and any(s in flowers for s in spref):
            w *= 1.10
        if savoid and any(s in flowers for s in savoid):
            w *= 0.90

        # Sympathy lily guard
        if edge_case == "sympathy" and "lily" in flowers:
            w *= 1.25

    # Iconic "only X" override intent
    if iconic_intent:
        if _is_mono(item) and iconic_intent in flowers:
            w *= 10.0
        elif iconic_intent in flowers:
            w *= 1.5
        else:
            w *= 0.6

    # Editorial base weight
    w *= max(0.1, float(item.get("weight", 50)) / 50.0)
    return w

def _apply_lg_policy(item: Dict[str, Any], emotion: str, edge_case: Optional[str]) -> float:
    if not item.get("luxury_grand"):
        return 1.0

    # Global block-list
    blocked = [x.lower() for x in TIER_POLICY.get("luxury_grand", {}).get("blocked_emotions", [])]
    if emotion.lower() in blocked:
        return 0.0

    # Edge soft policy
    if edge_case and edge_case in EDGE_CASE_KEYS:
        reg = EDGE_REGISTERS.get(edge_case, {})
        if str(reg.get("lg_policy", "")).lower() == "block":
            return 0.0
        return float(reg.get("lg_weight_multiplier", 1.0))

    return 1.0

# ------------------------------------------------------------
# Candidate building & picking
# ------------------------------------------------------------

def _candidates_for_tier(items: List[Dict[str, Any]], tier: str) -> List[Dict[str, Any]]:
    return [it for it in items if it.get("tier") == tier]

def _pick_best(cands: List[Dict[str, Any]], want_mono: Optional[bool]) -> Optional[Dict[str, Any]]:
    if not cands:
        return None
    filtered = cands
    if want_mono is True:
        filtered = [c for c in cands if _is_mono(c)]
    elif want_mono is False:
        filtered = [c for c in cands if not _is_mono(c)]
    if not filtered:
        filtered = cands
    return max(filtered, key=lambda x: x.get("_score", 0.0))

def _apply_substitution_notes(triad: List[Dict[str, Any]], redirect_from: Optional[str], alts: List[str]) -> None:
    """
    If user asked for an unavailable species (redirect_from), attach a note to each
    triad item whose flowers include one of the chosen substitutes in `alts`.
    Fallback: if nothing matches, put the note on the first MIX item only.
    """
    if not redirect_from or not alts:
        return
    note_tpl = SUB_NOTES.get(redirect_from.lower(), SUB_NOTES.get("default", "Requested {from} is seasonal/unavailable; offering {alt} as the nearest alternative."))
    matched = 0
    for it in triad:
        flowers = [f.lower() for f in (it.get("flowers") or [])]
        hit = next((a for a in alts if a.lower() in flowers), None)
        if hit:
            it["note"] = note_tpl.format(**{"from": redirect_from, "alt": hit})
            matched += 1
    if matched == 0:
        # fallback: first MIX only (keeps UI clean)
        for it in triad:
            if not it.get("mono"):
                it["note"] = note_tpl.format(**{"from": redirect_from, "alt": alts[0]})
                break

def _ensure_two_mix_one_mono(triad: List[Dict[str, Any]], all_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Guarantee exactly 2 MIX + 1 MONO while keeping tier order."""
    mixes = [i for i in triad if not _is_mono(i)]
    monos = [i for i in triad if _is_mono(i)]
    if len(mixes) == 2 and len(monos) == 1:
        return triad

    # Too many monos → downgrade extras to MIX by swapping with next best MIX from same tier pool (if exists)
    if len(monos) > 1:
        # Replace highest-rank mono (except keep at least one) with best MIX of the same tier
        to_replace = monos[1:]
        for m in to_replace:
            tier = m.get("tier")
            pool = [x for x in all_items if x.get("tier") == tier and not _is_mono(x)]
            pool = sorted(pool, key=lambda x: x.get("_score", 0.0), reverse=True)
            if pool:
                idx = triad.index(m)
                triad[idx] = pool[0]
        # recompute
        mixes = [i for i in triad if not _is_mono(i)]
        monos = [i for i in triad if _is_mono(i)]

    # If still wrong (e.g., 3 MIX), try to upgrade Luxury to MONO if available, else Signature
    if len(monos) == 0:
        for pref_tier in ["Luxury", "Signature", "Classic"]:
            pool = [x for x in all_items if x.get("tier") == pref_tier and _is_mono(x)]
            pool = sorted(pool, key=lambda x: x.get("_score", 0.0), reverse=True)
            if pool:
                # replace current item of that tier
                for i, it in enumerate(triad):
                    if it.get("tier") == pref_tier:
                        triad[i] = pool[0]
                        return triad
    return triad

# ------------------------------------------------------------
# Public entrypoint
# ------------------------------------------------------------

def selection_engine(prompt: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Returns exactly 3 curated items.
    - One per tier in order Classic → Signature → Luxury (LG sits within Luxury).
    - Exactly 2 MIX + 1 MONO.
    - Edge registers & notes applied.
    """
    context = context or {}
    p = normalize(prompt)

    # 1) read recent_ids at the top of curate/selection function
    recent_ids = set((context or {}).get("recent_ids", []) or [])
    original_catalog = list(CATALOG)

    # Emotion + edge-case
    emotion = detect_emotion(p, context)
    edge_case = detect_edge_register(p)

    # --- intent clarity & LG dampener ---
    bucket = (EMOTION_KEYWORDS.get("keywords", {}) or {}).get(emotion, [])
    matched_keywords = sum(1 for k in bucket if normalize(k) in normalize(prompt))
    clarity = _intent_clarity(prompt, matched_keywords)
    soft_lg_multiplier = 1.0
    if clarity == 0.0 and not _has_grand_intent(prompt):
        try:
            budget = int((context or {}).get("budget_inr", 0) or 0)
        except Exception:
            budget = 0
        if budget < 4999:
            soft_lg_multiplier = 0.8

    redirect_from: Optional[str] = None
    redirect_alts: List[str] = []

    # Check for specific substitution case (e.g., hydrangea)
    if "hydrangea" in p:
        redirect_from = "hydrangea"
        # Dummy alternatives for now; a real implementation would find these
        redirect_alts = ["lily", "chrysanthemum"]

    # Iconic override intent (only lilies/roses/orchids)
    iconic_intent = None
    m = ONLY_ICONIC_RE.search(p)
    if m:
        g = m.group(1).lower()
        if g.startswith("lil"):
            iconic_intent = "lily"
        elif g.startswith("rose"):
            iconic_intent = "rose"
        elif g.startswith("orchid"):
            iconic_intent = "orchid"

    # Register for edge-case
    register = EDGE_REGISTERS.get(edge_case or "", {})

    # Score all catalog items
    scored: List[Dict[str, Any]] = []
    # 2) when scoring/choosing, filter out recent_ids first
    candidates = [c for c in CATALOG if c["id"] not in recent_ids]
    
    # 3) if filtering leaves too few to satisfy 2 MIX + 1 MONO, fall back to the full set to keep guarantees
    if len(candidates) < 3:
        candidates = original_catalog

    for item in candidates:
        # Skip items missing required fields
        if item.get("tier") not in TIER_ORDER or not item.get("palette"):
            continue

        base = int(item.get("weight", 50))
        w = _compute_weight(item, base, register, edge_case, iconic_intent)
        w *= _apply_lg_policy(item, emotion, edge_case)

        # dampen LG softly for unclear intent (unless 'grand' or budget qualifies)
        if item.get("luxury_grand"):
            w *= soft_lg_multiplier

        # If LG blocked multiplier is 0 → effectively remove
        if w <= 0:
            continue

        # Add a tiny deterministic tie-break so identical inputs stay stable:
        if item.get("id"):
            w += (int(hashlib.sha256(item["id"].encode()).hexdigest(), 16) % 997) / 1e9

        candidate = dict(item)
        candidate["_score"] = w
        resolved_anchor = detect_emotion(prompt, context)
        # Strict edge gating + proper emotion (anchor) asisgnment
        candidate["emotion"]   = resolved_anchor
        candidate["edge_case"] = bool(edge_case in EDGE_CASE_KEYS)
        candidate["_edge_type"] = edge_case if edge_case in EDGE_CASE_KEYS else None
        scored.append(candidate)

    if not scored:
        return []

    # Pick per tier with preference: MIX for Classic/Signature, any for Luxury (mono often chosen last)
    triad: List[Dict[str, Any]] = []
    for tier in TIER_ORDER:
        pool = _candidates_for_tier(scored, tier)
        want_mono = None
        if tier in ("Classic", "Signature"):
            want_mono = False
        best = _pick_best(pool, want_mono)
        if best:
            triad.append(best)

    # If we failed to get three (catalog scarcity), fill from next best irrespective of tier uniqueness
    # But keep only unique tiers (strong invariant)
    tiers_present = {it.get("tier") for it in triad}
    for tier in TIER_ORDER:
        if tier not in tiers_present:
            pool = _candidates_for_tier(scored, tier)
            if pool:
                triad.append(pool[0])
                tiers_present.add(tier)

    # Ensure we have exactly one per tier (drop extras if any weirdness)
    triad = _order_by_tier({t: max([i for i in triad if i.get("tier")==t], key=lambda x: x.get("_score", 0.0)) for t in TIER_ORDER if any(i.get("tier")==t for i in triad)}.values())  # type: ignore

    # Strong iconic override: try to replace the mono slot with the iconic species mono when present
    if iconic_intent:
        # find best mono of species; prefer Luxury, then Signature, then Classic
        for pref_tier in ["Luxury", "Signature", "Classic"]:
            pool = [x for x in scored if x.get("tier")==pref_tier and _is_mono(x) and _has_species(x, iconic_intent)]
            pool = sorted(pool, key=lambda x: x.get("_score", 0.0), reverse=True)
            if pool:
                # replace the current item for that pref_tier
                for i, it in enumerate(triad):
                    if it.get("tier") == pref_tier:
                        triad[i] = pool[0]
                        break
                break

    # Last-mile ritual: exactly 2 MIX + 1 MONO
    triad = _ensure_two_mix_one_mono(triad, scored)

    # Apply notes for substituted species
    _apply_substitution_notes(triad, redirect_from, redirect_alts)

    # if the user intent is very generic, add a single clarifying note on first MIX
    if clarity == 0.0:
        _add_unclear_mix_note(triad)

    # Edge-case copy limit
    for it in triad:
        it["desc"] = _enforce_copy_limit(it.get("desc", ""), it.get("_edge_type"))

    # Map fields for API output
    out = []
    for it in _order_by_tier(triad):
        out.append({
            "id": it.get("id"),
            "title": it.get("title"),
            "desc": it.get("desc"),
            "image": it.get("image_url") or it.get("image"),
            "price": it.get("price_inr"),
            "currency": "INR",
            "emotion": it.get("emotion"),
            "tier": it.get("tier"),
            "packaging": it.get("packaging"),
            "mono": bool(it.get("mono")),
            "palette": list(it.get("palette") or []),
            "luxury_grand": bool(it.get("luxury_grand")),
            "note": it.get("note"),
            "edge_case": bool(it.get("edge_case")),
            "edge_type": it.get("_edge_type"), # optional exposure
        })
    return out

# Back-compat thin wrapper
def curate(prompt: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    return selection_engine(prompt, context or {})

# FastAPI endpoints
app = FastAPI()

@app.post("/curate", tags=["public"])
async def curate_post(q: Optional[str] = Query(None), prompt: Optional[str] = Query(None), recent_ids: Optional[List[str]] = Query(None)):
    text = (q or prompt or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Invalid input")
    context = {"recent_ids": recent_ids}
    items = selection_engine(text, context)
    return {"items": items, "edge_case": any(i.get("edge_case") for i in items)}

# Alias the existing endpoints to also work with the /api prefix
@app.post("/api/curate", tags=["public"])
async def curate_post_api(q: Optional[str] = Query(None), prompt: Optional[str] = Query(None), recent_ids: Optional[List[str]] = Query(None)):
    return await curate_post(q, prompt, recent_ids)

@app.get("/curate", tags=["public"])
async def curate_get(q: Optional[str] = Query(None), prompt: Optional[str] = Query(None), recent_ids: Optional[List[str]] = Query(None)):
    text = (q or prompt or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Invalid input")
    context = {"recent_ids": recent_ids}
    items = selection_engine(text, context)
    return {"items": items, "edge_case": any(i.get("edge_case") for i in items)}

@app.get("/api/curate", tags=["public"])
async def curate_get_api(q: Optional[str] = Query(None), prompt: Optional[str] = Query(None), recent_ids: Optional[List[str]] = Query(None)):
    return await curate_get(q, prompt, recent_ids)
