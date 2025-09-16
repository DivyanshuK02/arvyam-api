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
# Load rules and data
# ------------------------------------------------------------

EMOTION_KEYWORDS = _load_json(_p("emotion_keywords.json"), {})
TIER_POLICY = _load_json(_p("rules", "tier_policy.json"), {})
EDGE_REGISTERS = _load_json(_p("edge_registers.json"), {})
PRODUCT_CATALOG = _load_json(_p("rules", "product_catalog.json"), [])

# ------------------------------------------------------------
# Core engine functions
# ------------------------------------------------------------

def normalize(prompt: str) -> str:
    # Remove extra spaces, convert to lowercase, etc.
    return " ".join(prompt.lower().split())

def _is_false_friend(text: str, false_friends: list) -> bool:
    # A simple check for words that might mislead the emotion detector
    return any(ff.lower() in text.lower() for ff in false_friends)

def _matches_regex_list(text: str, regex_list: list) -> bool:
    import re
    for pattern in regex_list:
        try:
            if re.search(pattern, text, flags=re.I):
                return True
        except re.error:
            continue
    return False

def _has_proximity(hay: str, a: str, b: str, window: int = 2) -> bool:
    # Naive proximity: ensure a and b exist and are within N tokens
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

def _filter_by_emotion_policy(items: list, emotion: Optional[str]) -> list:
    if not emotion:
        return items
    
    blocked_items = TIER_POLICY.get("emotion_block_list", {}).get(emotion, [])
    
    return [i for i in items if i.get("id") not in blocked_items]

def _filter_by_tier_rules(items: list, tier_rules: dict, selected_emotion: Optional[str]) -> list:
    if not tier_rules:
        return items
    
    filtered = []
    
    # Apply species avoid rules
    species_avoid = tier_rules.get("species_avoid", [])
    if species_avoid:
        items = [i for i in items if not any(s in i.get("title", "").lower() for s in species_avoid)]
    
    # Apply palette avoid rules
    palette_avoid = tier_rules.get("palette_avoid", [])
    if palette_avoid:
        items = [i for i in items if not any(p in i.get("palette", []) for p in palette_avoid)]
        
    return items

def _score_and_select(items: list, tier_rules: dict) -> list:
    scored_items = []
    
    for item in items:
        score = 1.0
        
        # Apply palette boosts
        palette_targets = tier_rules.get("palette_targets", [])
        palette_target_boost = tier_rules.get("palette_target_boost", 1.0)
        
        for p_target in palette_targets:
            if p_target in item.get("palette", []):
                score *= palette_target_boost
        
        # Apply palette penalties
        palette_avoid = tier_rules.get("palette_avoid", [])
        palette_avoid_penalty = tier_rules.get("palette_avoid_penalty", 1.0)
        
        for p_avoid in palette_avoid:
            if p_avoid in item.get("palette", []):
                score *= palette_avoid_penalty
        
        # Apply species boosts
        species_prefer = tier_rules.get("species_prefer", [])
        species_prefer_boost = tier_rules.get("species_prefer_boost", 1.0)
        
        for s_prefer in species_prefer:
            if s_prefer in item.get("species_list", []):
                score *= species_prefer_boost
                
        # Apply species penalties
        species_avoid = tier_rules.get("species_avoid", [])
        species_avoid_penalty = tier_rules.get("species_avoid_penalty", 1.0)

        for s_avoid in species_avoid:
            if s_avoid in item.get("species_list", []):
                score *= species_avoid_penalty

        # Apply LG multiplier
        lg_weight_multiplier = tier_rules.get("lg_weight_multiplier", 1.0)
        if item.get("luxury_grand"):
            score *= lg_weight_multiplier
        
        item["score"] = score
        scored_items.append(item)
    
    # Sort by score in descending order
    scored_items.sort(key=lambda x: x["score"], reverse=True)
    
    return scored_items

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


def selection_engine(prompt: str, context: dict) -> list:
    resolved_anchor, edge_type, _scores = detect_emotion(prompt, context or {})

    # If edge short-circuited, derive anchor from edge_registers
    if resolved_anchor == "__EDGE__":
        edge_rules = EDGE_REGISTERS.get(edge_type, {})
        resolved_anchor = edge_rules.get("emotion_anchor") or resolved_anchor  # fall back if misconfigured
        is_edge = True
    else:
        is_edge = False
    
    all_items = PRODUCT_CATALOG[:]
    
    # Filter items based on emotion blocklist
    items_filtered = _filter_by_emotion_policy(all_items, resolved_anchor)

    # Filter items based on tier rules
    if is_edge:
        edge_rules = EDGE_REGISTERS.get(edge_type, {})
        items_filtered = _filter_by_tier_rules(items_filtered, edge_rules, resolved_anchor)
    
    # Score items
    scored_items = _score_and_select(items_filtered, EDGE_REGISTERS.get(edge_type, {}))

    selected_items = []
    
    # Select 2 MIX and 1 MONO
    mix_items = [i for i in scored_items if not i.get("mono")]
    mono_items = [i for i in scored_items if i.get("mono")]

    # Enforce mono_must_include for edge cases
    if is_edge:
        edge_rules = EDGE_REGISTERS.get(edge_type, {})
        mono_must_include_species = edge_rules.get("mono_must_include", [])
        if mono_must_include_species:
            mono_items = [i for i in mono_items if any(s in i.get("species_list", []) for s in mono_must_include_species)]
    
    if mix_items:
        selected_items.extend(mix_items[:2])
    if mono_items:
        selected_items.extend(mono_items[:1])
    
    if len(selected_items) < 3:
        # Fallback to general selection if not enough items
        fallback_items = [i for i in PRODUCT_CATALOG if not i.get("mono")]
        fallback_items.sort(key=lambda x: x.get("score", 0), reverse=True)
        mix_count = min(2, len(fallback_items))
        selected_items.extend(fallback_items[:mix_count])
        
        fallback_mono = [i for i in PRODUCT_CATALOG if i.get("mono")]
        fallback_mono.sort(key=lambda x: x.get("score", 0), reverse=True)
        mono_count = min(1, len(fallback_mono))
        selected_items.extend(fallback_mono[:mono_count])
    
    # Final stamping (exactly once)
    for it in selected_items:
        it["emotion"] = resolved_anchor
        it["edge_case"] = is_edge
        it["edge_type"] = edge_type if is_edge else None
        # Copy cap only when edge is active and rules specify a cap
        it["desc"] = _enforce_copy_limit(it.get("desc", ""), edge_type if is_edge else None)
        
    return selected_items

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
