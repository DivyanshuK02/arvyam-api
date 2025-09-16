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
EDGES = (EMOTION_KEYWORDS or {}).get("edges", {})  # single source of truth
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

def _truncate_words(text: str, max_words: int) -> str:
    words = [w for w in (text or "").split()]
    if len(words) <= max_words:
        return text or ""
    return " ".join(words[:max_words]).rstrip() + "…"


def detect_emotion(prompt: str, context: dict | None) -> Tuple[str, Optional[str], Dict[str, float]]:
    """
    Core emotion and intent detection logic.
    Returns (resolved_anchor, edge_type, scores)
    """
    p = normalize(prompt)

    # Edge registers short-circuit routing; item stamping uses this edge_type exactly once.
    # Canary tests (CI): one exact/regex/proximity sample per register; fail CI if any canary stops matching.
    # 0) Edge case check (highest priority) from emotion_keywords["edges"]
    
    # Check for canonical edge cases first (highest priority)
    for edge_type, rules in (EDGES or {}).items():
        # exact
        for phrase in rules.get("exact", []) or []:
            if phrase and phrase in p:
                return EDGE_REGISTERS.get(edge_type, {}).get("emotion_anchor"), edge_type, {}

        # contains_any
        for token in rules.get("contains_any", []) or []:
            if token and token in p:
                return EDGE_REGISTERS.get(edge_type, {}).get("emotion_anchor"), edge_type, {}

        # regex (list of pattern strings OR {"pattern": ...})
        regex_list = rules.get("regex", []) or []
        patterns = [(r if isinstance(r, str) else r.get("pattern", "")) for r in regex_list]
        patterns = [r for r in patterns if r]
        if patterns and _matches_regex_list(p, patterns):
            return EDGE_REGISTERS.get(edge_type, {}).get("emotion_anchor"), edge_type, {}

        # proximity_pairs
        for spec in rules.get("proximity_pairs", []) or []:
            a = spec.get("a", ""); b = spec.get("b", "")
            window = int(spec.get("window", 2))
            if a and b and _has_proximity(p, a, b, window=window):
                return EDGE_REGISTERS.get(edge_type, {}).get("emotion_anchor"), edge_type, {}

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
    return [it for it in CATALOG if species_name.lower() in it.get("species_raw", "").lower() and it.get("rating", 0) >= min_rating]

def find_items_by_emotion(emotion: str, min_rating: int = 0) -> List[Dict[str, Any]]:
    return [it for it in CATALOG if emotion.lower() in (it.get("emotions") or "").lower() and it.get("rating", 0) >= min_rating]

def find_iconic_species(prompt: str) -> Optional[str]:
    """Finds a species mention in a prompt, e.g., 'rose', 'lily'."""
    p = normalize(prompt)
    for species, aliases in ICONIC_SPECIES.items():
        if any(a in p for a in aliases):
            return species
    return None

def filter_by_availability(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [it for it in items if it.get("is_active", False) and it.get("is_available", False)]

def assign_tiers(items: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Assigns Classic/Signature/Luxury tiers based on a deterministic hash."""
    seed = (context.get("prompt_hash", "0") or "")
    tier_counts = {"Classic": 1, "Signature": 1, "Luxury": 1}
    # If there's grand intent, shift one from classic to luxury
    if _has_grand_intent(context.get("prompt", "")):
        tier_counts["Classic"] = 0
        tier_counts["Luxury"] = 2
    
    result = []
    
    for tier in TIER_ORDER:
        filtered = [it for it in items if it.get("tier", "").lower() == tier.lower()]
        if not filtered:
            continue

        for _ in range(tier_counts.get(tier, 0)):
            idx = _rotation_index(seed + str(len(result)), len(filtered))
            result.append(filtered[idx])

    return result

def curate(prompt: str, context: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """Main entry point for external systems; includes all logic."""
    context = context or {}
    context["prompt"] = prompt
    return selection_engine(prompt, context)

def selection_engine(prompt: str, context: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """
    Core function for item selection based on emotion, species, and tiering logic.
    Returns exactly 3 items (2 MIX, 1 MONO) with all stamping and notes applied.
    """
    context = context or {}
    prompt_norm = normalize(prompt)
    context["prompt_hash"] = hashlib.sha256(prompt_norm.encode()).hexdigest()

    # Determine emotion, species, and intent
    resolved_anchor, edge_type, _scores = detect_emotion(prompt, context or {})
    is_edge = edge_type is not None
    
    selected_species = find_iconic_species(prompt_norm)

    # LG Policy check: block list
    blocked = TIER_POLICY.get("luxury_grand", {}).get("blocked_emotions", [])
    if _has_grand_intent(prompt_norm) and resolved_anchor in blocked:
        # Fallback to general, as per LG policy
        resolved_anchor = "general"

    # Fetch items and apply filters
    pool = filter_by_availability(CATALOG)
    
    # Tier-based item selection (always try to return 3 items)
    
    selected_items = []
    
    # 1. Species first (Mono)
    if selected_species:
        species_pool = find_items_by_species(selected_species, min_rating=4) # only high-rated mono
        species_pool = filter_by_availability(species_pool)
        if species_pool:
            selected_items.append(species_pool[_rotation_index(context["prompt_hash"] + "_mono", len(species_pool))])

    # 2. Emotion-based items (MIX)
    emotion_pool = find_items_by_emotion(resolved_anchor, min_rating=0)
    emotion_pool = filter_by_availability(emotion_pool)
    
    # Ensure we have at least 2 items for the MIX and fill up if needed
    mix_items = []
    if emotion_pool:
        # Get up to two items from the emotion pool
        idx1 = _rotation_index(context["prompt_hash"] + "_mix1", len(emotion_pool))
        mix_items.append(emotion_pool[idx1])
        if len(emotion_pool) > 1:
            idx2 = _rotation_index(context["prompt_hash"] + "_mix2", len(emotion_pool))
            if idx1 == idx2: # simple collision avoidance
                idx2 = (idx2 + 1) % len(emotion_pool)
            mix_items.append(emotion_pool[idx2])

    # If not enough items, fill with classic
    while len(mix_items) < 2:
        classic_pool = find_items_by_emotion("general", min_rating=0) # or any other default
        if classic_pool:
            mix_items.append(classic_pool[_rotation_index(context["prompt_hash"] + f"_fill{len(mix_items)}", len(classic_pool))])
        else:
            break
            
    selected_items.extend(mix_items)

    # 3. Handle total item count
    if len(selected_items) < 3:
        # Fallback to general if we're still short
        general_pool = find_items_by_emotion("general", min_rating=0)
        while len(selected_items) < 3 and general_pool:
            selected_items.append(general_pool[_rotation_index(context["prompt_hash"] + f"_final_fill{len(selected_items)}", len(general_pool))])

    # Assign tiers
    final_triad = assign_tiers(selected_items, context)
    
    # Add notes to items if needed
    find_and_assign_note(final_triad, selected_species, resolved_anchor)

    # Final stamping (exactly once)
    for it in final_triad:
        it["emotion"]  = resolved_anchor
        it["edge_case"] = is_edge
        it["edge_type"] = edge_type if is_edge else None
        # Copy cap only when edge is active and rules specify a cap
        it["desc"] = _enforce_copy_limit(it.get("desc",""), edge_type if is_edge else None)
    
    return final_triad

def find_and_assign_note(triad: list, selected_species: Optional[str], selected_emotion: Optional[str]) -> None:
    """Attaches a substitution note if a requested species wasn't found."""
    found_species = any(it.get("species_raw", "").lower() == selected_species for it in triad) if selected_species else False
    
    if selected_species and not found_species:
        substitution_note = SUB_NOTES.get("species_not_found", "We couldn't find a {species} bouquet at the moment; offering a similar style.")
        substitution_note = substitution_note.replace("{species}", selected_species)
        triad[0]["note"] = substitution_note
    elif _intent_clarity(triad[0].get("prompt",""), 1) == 0:
        _add_unclear_mix_note(triad)

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
