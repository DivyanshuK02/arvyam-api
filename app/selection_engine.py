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
EDGES = (EMOTION_KEYWORDS or {}).get("edges", {})
EDGE_REGISTERS = _load_json(os.path.join(RULES_DIR, "edge_registers.json"), {})
TIER_POLICY = _load_json(os.path.join(RULES_DIR, "tier_policy.json"), {"luxury_grand": {"blocked_emotions": []}})
SUB_NOTES = _load_json(os.path.join(RULES_DIR, "substitution_notes.json"), {"default": "Requested {from} is seasonal/unavailable; offering {alt} as the nearest alternative."})
ANCHOR_THRESHOLDS = _load_json(os.path.join(RULES_DIR, "anchor_thresholds.json"), {})
# Edge register keys (strict): only these four are considered "edge cases".
EDGE_CASE_KEYS = {"sympathy", "apology", "farewell", "valentine"}
FEATURE_MULTI_ANCHOR_LOGGING = os.getenv("FEATURE_MULTI_ANCHOR_LOGGING", "0") == "1" # Off by default

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

def _stable_hash_u32(s: str) -> int:
    h = hashlib.sha256((s or "").encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def _rotation_index(seed: str, k: int) -> int:
    if k <= 1: return 0
    return _stable_hash_u32(seed) % k

def detect_edge_register_from_emotion_keywords(
    prompt_norm: str,
    emotion_keywords: dict
) -> Optional[str]:
    """
    Single source of truth for canonical edge detection.
    Reads enriched rules from emotion_keywords["edges"][edge_type].
    Returns: edge_type ("sympathy" | "apology" | "farewell" | "valentine") or None.
    """
    edges = (emotion_keywords or {}).get("edges", {})
    if not isinstance(edges, dict):
        return None

    def any_contains(hay: str, needles: list[str]) -> bool:
        return any(n for n in needles if isinstance(n, str) and n in hay)

    def matches_regex_list(hay: str, regex_specs: list[dict]) -> bool:
        import re
        for spec in regex_specs:
            pat = spec.get("pattern")
            if not pat:
                continue
            try:
                if re.search(pat, hay, flags=re.I):
                    return True
            except re.error:
                continue
        return False

    def has_proximity(hay: str, a: str, b: str, window: int = 2) -> bool:
        # naive proximity: ensure a and b exist and are within N tokens
        if not a or not b:
            return False
        toks = hay.split()
        try:
            idxs_a = [i for i, t in enumerate(toks) if a in t]
            idxs_b = [i for i, t in enumerate(toks) if b in t]
        except Exception:
            return False
        for ia in idxs_a:
            for ib in idxs_b:
                if abs(ia - ib) <= window:
                    return True
        return False

    # Priority order inside each edge: exact → contains_any → regex → proximity_pairs
    for edge_type, enr in edges.items():
        if not isinstance(enr, dict):
            continue

        # exact
        exact = enr.get("exact", [])
        if isinstance(exact, list) and any_contains(prompt_norm, [s for s in exact if isinstance(s, str)]):
            return edge_type

        # contains_any
        contains_any = enr.get("contains_any", [])
        if isinstance(contains_any, list) and any_contains(prompt_norm, [s for s in contains_any if isinstance(s, str)]):
            return edge_type

        # regex  (>>> this is where the missing for-loop caused NameError)
        regex_specs = enr.get("regex", [])
        if isinstance(regex_specs, list) and matches_regex_list(prompt_norm, regex_specs):
            return edge_type

        # proximity_pairs (list of {"a": "...", "b": "...", "window": 2})
        proximity_specs = enr.get("proximity_pairs", [])
        if isinstance(proximity_specs, list):
            for spec in proximity_specs:
                if not isinstance(spec, dict):
                    continue
                a = spec.get("a", "")
                b = spec.get("b", "")
                w = int(spec.get("window", 2))
                if has_proximity(prompt_norm, a, b, w):
                    return edge_type

    return None

def detect_emotion(prompt: str, context: dict | None) -> Tuple[str, Optional[str], Dict[str, float]]:
    p = normalize(prompt)

    # Edge registers short-circuit routing; item stamping uses this edge_type exactly once.
    # Canary tests (CI): one exact/regex/proximity sample per register; fail CI if any canary stops matching.
    # 0) Edge case check (highest priority) from emotion_keywords["edges"]
    edge_type = detect_edge_register_from_emotion_keywords(p, EMOTION_KEYWORDS)
    if edge_type:
        # Return early with the canonical edge_type; anchor will be resolved later from edge_registers.
        return ("__EDGE__", edge_type, {})
    
    # 1) Safe context hint (only accept known anchors)
    anchors: List[str] = EMOTION_KEYWORDS.get("anchors", []) or []
    hint = (context or {}).get("emotion_hint", "") or ""
    hint = hint.strip()
    if hint and hint in anchors:
        return hint, None, {}
    
    # Tables
    exact_map = EMOTION_KEYWORDS.get("exact", {}) or {}
    combos = EMOTION_KEYWORDS.get("combos", []) or {}
    buckets = EMOTION_KEYWORDS.get("keywords", {}) or {}
    disamb = EMOTION_KEYWORDS.get("disambiguation", []) or {}

    # 2) Disambiguation rules (highest precision)
    for rule in disamb:
        when_any = rule.get("when_any", [])
        if not when_any or not any(w in p for w in map(str.lower, when_any)):
            continue
        for branch in rule.get("route", []):
            if_any = branch.get("if_any", [])
            if if_any and any(t in p for t in map(str.lower, if_any)):
                return branch.get("anchor"), None, {}
            if "else" in branch:
                return branch["else"], None, {}

    # 3) Exact phrase map (case-insensitive contains)
    for phrase, anchor in exact_map.items():
        if phrase.lower() in p:
            return anchor, None, {}

    # 4) Combos: all terms present
    for combo in combos:
        terms = [t.lower() for t in combo.get("all", [])]
        if terms and all(t in p for t in terms):
            return combo.get("anchor"), None, {}

    # 5) Enriched (optional) — regex & proximity_pairs
    enr = EMOTION_KEYWORDS.get("enriched", {}) or {}
    if enr.get("false_friends") and _is_false_friend(p, enr["false_friends"]): # fall through to buckets without enriched scoring
        pass
    else:
        # regex
        for spec in enr.get("regex", []):
            if _matches_regex_list(p, [spec.get("pattern","")]):
                return spec.get("anchor"), None, {}

        # proximity pairs
        for spec in enr.get("proximity_pairs", []):
            if _has_proximity(p, spec.get("a",""), spec.get("b",""), window=int(spec.get("window",2))):
                return spec.get("anchor"), None, {}
        
    # 6) Buckets: score by keyword hits; tie-break by anchors[] order
    scores = {a: 0 for a in anchors}
    for a, words in (buckets or {}).items():
        scores[a] = sum(1 for w in (words or []) if w.lower() in p)
    # pick best; if tie, prefer earlier in anchors[]
    best = max(anchors, key=lambda a: (scores.get(a,0), -anchors.index(a))) if anchors else None
    if best and scores.get(best, 0) > 0: # Normalize scores for logging
        max_score = scores.get(best) or 1
        scores_norm = {a: s / max_score for a, s in scores.items()}
        return best, None, scores_norm

    # 7) Fallback to an empty or default emotion
    if not p:
        return "empty", None, {}
    
    return "general", None, {}


def find_items_by_species(species_name: str, min_rating: int = 0) -> List[Dict[str, Any]]:
    # ... (rest of the code below this line is truncated for brevity)
    pass

def find_items_by_emotion(emotion: str, min_rating: int = 0) -> List[Dict[str, Any]]:
    # ...
    pass

def find_iconic_species(prompt: str) -> Optional[str]:
    # ...
    pass

def filter_by_availability(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # ...
    pass

def assign_tiers(items: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    # ...
    pass

def curate(prompt: str, context: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    # ...
    pass

def selection_engine(prompt: str, context: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    # ...
    resolved_anchor, edge_type, _scores = detect_emotion(prompt, context or {})

    # If edge short-circuited, derive anchor from edge_registers
    if resolved_anchor == "__EDGE__":
        edge_rules = EDGE_REGISTERS.get(edge_type, {})
        resolved_anchor = edge_rules.get("emotion_anchor") or resolved_anchor  # fall back if misconfigured
        is_edge = True
    else:
        is_edge = False
    
    # ... your scoring/selection happens here, yielding selected_items (2 MIX + 1 MONO)
    selected_items = []

    # Final stamping (exactly once)
    for it in selected_items:
        it["emotion"]  = resolved_anchor
        it["edge_case"] = is_edge
        it["edge_type"] = edge_type if is_edge else None
        # Copy cap only when edge is active and rules specify a cap
        it["desc"] = _enforce_copy_limit(it.get("desc",""), edge_type if is_edge else None)
    
    return selected_items

def find_and_assign_note(triad: list, selected_species: Optional[str], selected_emotion: Optional[str]) -> None:
    # ...
    pass

def _enforce_copy_limit(text: str, edge_type: Optional[str]) -> str:
    """
    Truncates `text` to the copy_max_words for the given edge_type.
    Pass the register key (e.g., "sympathy", "apology", "farewell", "valentine") or None.
    Caps to Phase-1 hard fence (≤20 words). Called only when an edge case is active.
    """
    max_words = 20  # hard fence
    if edge_type:
        regs = EDGE_REGISTERS.get(edge_type, {})
        cap = int(regs.get("copy_max_words", max_words))
        max_words = min(max_words, cap)

    words = [w for w in (text or "").split()]
    if len(words) <= max_words:
        return text or ""
    return " ".join(words[:max_words]).rstrip() + "…"

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
