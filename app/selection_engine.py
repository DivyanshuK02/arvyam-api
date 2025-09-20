# app/selection_engine.py
# P1.4a/PR-2: This file has been refactored to be a pure logic module.
# All FastAPI routes and web-server code have been removed.
# The main selection_engine function now returns a (items, context, meta) tuple.
# The core selection algorithm has been preserved.

from __future__ import annotations

import json, os, re, hashlib, logging
from typing import Any, Dict, List, Optional, Tuple
from fastapi import HTTPException # Kept for internal error signaling
from collections import defaultdict

__all__ = ["selection_engine", "normalize", "detect_emotion", "_transform_for_api"]

# ------------------------------------------------------------
# File helpers
# ------------------------------------------------------------

ROOT = os.path.dirname(__file__)
log = logging.getLogger("arvyam.engine")

def _p(*parts: str) -> str:
    return os.path.join(ROOT, *parts)

def _load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        log.warning(f"Failed to load {path}: {e}")
        return default

# ------------------------------------------------------------
# Data sources (all project-owned JSON files)
# ------------------------------------------------------------

CATALOG = _load_json(_p("catalog.json"), [])
CAT_BY_ID = {item['id']: item for item in CATALOG if 'id' in item}
RULES_DIR = _p("rules")
EMOTION_KEYWORDS = _load_json(os.path.join(RULES_DIR, "emotion_keywords.json"), {})
EDGES = (EMOTION_KEYWORDS.get("edges", {}) or {})
EDGE_REGISTERS = _load_json(os.path.join(RULES_DIR, "edge_registers.json"), {})
TIER_POLICY = _load_json(os.path.join(RULES_DIR, "tier_policy.json"), {"luxury_grand": {"blocked_emotions": []}})
SUB_NOTES = _load_json(os.path.join(RULES_DIR, "substitution_notes.json"), {"default": "Requested {from} is seasonal/unavailable; offering {alt} as the nearest alternative."})
ANCHOR_THRESHOLDS = _load_json(os.path.join(RULES_DIR, "anchor_thresholds.json"), {})
SENTIMENT_FAMILIES = _load_json(os.path.join(RULES_DIR, "sentiment_families.json"), {})

EDGE_CASE_KEYS = {"sympathy", "apology", "farewell", "valentine"}
FEATURE_MULTI_ANCHOR_LOGGING = os.getenv("FEATURE_MULTI_ANCHOR_LOGGING", "0") == "1" # Off by default

def _validate_catalog_on_startup():
    """Validate catalog data at startup to prevent runtime failures"""
    if not CATALOG:
        log.critical("CATALOG is empty - service cannot start")
        raise RuntimeError("Invalid catalog configuration")
    
    if not isinstance(CATALOG, list):
        log.critical("CATALOG must be a list")
        raise RuntimeError("Invalid catalog format")
    
    for i, item in enumerate(CATALOG):
        if not isinstance(item, dict):
            log.critical(f"Catalog item {i} is not a dictionary")
            raise RuntimeError("Invalid catalog item format")
        
        has_image = bool(item.get("image") or item.get("image_url"))
        has_price = bool(item.get("price") is not None or item.get("price_inr") is not None)
        
        if not item.get("id"):
            log.critical(f"Catalog item {i} missing 'id' field")
            raise RuntimeError("Invalid catalog schema: missing id")
            
        if not has_image:
            log.warning(f"Catalog item {item.get('id')} missing image field")
            
        if not has_price:
            log.warning(f"Catalog item {item.get('id')} missing price field")

    log.info(f"Catalog validation passed: {len(CATALOG)} items loaded")

def _validate_catalog_families_on_startup():
    """PHASE-2: Validate catalog items against sentiment families config."""
    if not SENTIMENT_FAMILIES.get("sentiment_families"):
        log.warning("Sentiment families config is missing or empty.")
        return

    all_families = SENTIMENT_FAMILIES["sentiment_families"]
    emotion_to_family = {}
    for fname, fconfig in all_families.items():
        if "subfamilies" in fconfig:
            for sub_name, sub_config in fconfig["subfamilies"].items():
                for emotion in sub_config.get("emotions", []):
                    emotion_to_family[emotion] = sub_name
        else:
            for emotion in fconfig.get("emotions", []):
                emotion_to_family[emotion] = fname
    
    for item in CATALOG:
        item_id = item.get("id", "Unknown")
        family = item.get("sentiment_family")
        emotion = item.get("emotion")
        if not family:
            log.warning(f"Catalog item {item_id} is missing 'sentiment_family'. Treating as UNSPECIFIED.")
        if emotion and emotion not in emotion_to_family:
            log.warning(f"Catalog item {item_id} has emotion '{emotion}' which does not map to any sentiment family.")

# Validate on module import
_validate_catalog_on_startup()
_validate_catalog_families_on_startup()

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

TIER_ORDER = ["Classic", "Signature", "Luxury"]
TIER_RANK = {t: i for i, t in enumerate(TIER_ORDER)}

ICONIC_SPECIES = { "lily": ["lily", "lilies"], "rose": ["rose", "roses"], "orchid": ["orchid", "orchids"] }
ONLY_ICONIC_RE = re.compile(r"\bonly\s+(lil(?:y|ies)|roses?|orchids?)\b", re.I)

VAGUE_TERMS = {"nice", "beautiful", "pretty", "some", "any", "simple", "best", "good", "flowers", "flower"}
GRAND_INTENT = {"grand","bigger","large","lavish","extravagant","50+","hundred","massive","most beautiful"}

def normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s[:500]

def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z]+", normalize(s))

def _intent_clarity(prompt: str, matched_keywords: int) -> float:
    p = normalize(prompt)
    words = re.findall(r"[a-z]+", p)
    if matched_keywords == 0 and (len(words) <= 4 or all(w in VAGUE_TERMS for w in words)):
        return 0.0
    return 1.0

def _has_grand_intent(prompt: str) -> bool:
    p = normalize(prompt)
    return any(w in p for w in GRAND_INTENT)

def _add_unclear_mix_note(triad: list) -> None:
    for it in triad:
        if not it.get("mono") and not it.get("note"):
            it["note"] = "Versatile picks while you decide - tell us the occasion for a more personal curation."
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

def _stable_hash_u32(s: str) -> int:
    h = hashlib.sha256((s or "").encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def _rotation_index(seed: str, k: int) -> int:
    if k <= 1: return 0
    return _stable_hash_u32(seed) % k

def _suppress_recent(items: list[dict], recent_set: set[str]) -> list[dict]:
    if not recent_set:
        return items
    return [it for it in items if _stable_id(it) not in recent_set]

def _has_emotion_match(item: Dict[str, Any], emotion: str) -> bool:
    target = (emotion or "").lower()
    v = item.get("emotion")
    if isinstance(v, str) and target in v.lower():
        return True
    v = item.get("emotions")
    if isinstance(v, str) and target in v.lower():
        return True
    if isinstance(v, list) and any(target == (x or "").lower() for x in v):
        return True
    return False

def _match_edge(text: str, rules: dict) -> bool:
    if not rules: return False
    for phrase in (rules.get("exact") or []):
        if phrase and phrase.lower() in text: return True
    if any(tok and tok.lower() in text for tok in (rules.get("contains_any") or [])): return True
    patterns = [r if isinstance(r, str) else r.get("pattern", "") for r in (rules.get("regex") or [])]
    if patterns and _matches_regex_list(text, patterns): return True
    for spec in (rules.get("proximity_pairs") or []):
        if _has_proximity(text, spec.get("a",""), spec.get("b",""), int(spec.get("window",2))): return True
    return False

def _stable_id(it: dict) -> str:
    return str(it.get("id", "unknown"))

def _get_target_family(resolved_anchor: str, edge_type: Optional[str], relationship_context: Optional[str] = None) -> str:
    if edge_type:
        reg = EDGE_REGISTERS.get(edge_type)
        if reg:
            if edge_type == "apology" and relationship_context:
                subfamily_map = { "romantic": "romantic_repair", "familial": "familial_repair", "professional": "professional_repair", "friendship": "friendship_repair", }
                return subfamily_map.get(relationship_context, "friendship_repair")
            return reg.get("sentiment_family", "UNSPECIFIED")
    all_families = SENTIMENT_FAMILIES.get("sentiment_families", {})
    for fname, fconfig in all_families.items():
        if "subfamilies" in fconfig:
            for sub_name, sub_config in fconfig["subfamilies"].items():
                if resolved_anchor in sub_config.get("emotions", []):
                    return sub_name
        elif resolved_anchor in fconfig.get("emotions", []):
            return fname
    return "UNSPECIFIED"

def _get_family_config(target_family: str) -> dict:
    all_families = SENTIMENT_FAMILIES.get("sentiment_families", {})
    for fname, fconfig in all_families.items():
        if fname == target_family:
            return fconfig
        if "subfamilies" in fconfig:
            if target_family in fconfig["subfamilies"]:
                return fconfig["subfamilies"][target_family]
    return {}

def _filter_by_family(pool: list[dict], target_family: str, context: dict) -> list[dict]:
    if not target_family or target_family == "UNSPECIFIED":
        return pool
    family_config = _get_family_config(target_family)
    barriers = set(family_config.get("contamination_barriers", []))
    context["barriers_triggered"] = list(barriers)
    def is_allowed(item: dict) -> bool:
        item_family = item.get("sentiment_family") or "UNSPECIFIED"
        if item_family in barriers: return False
        return item_family == target_family or item_family == "UNSPECIFIED"
    return [it for it in pool if is_allowed(it)]

def _ensure_one_mono_in_triad(triad: list[dict], context: dict) -> list[dict]:
    if not triad or len(triad) != 3:
        return triad
    mono_items = [it for it in triad if it.get("mono")]
    if len(mono_items) == 1:
        return triad
    if len(mono_items) > 1:
        first_mono_found = False
        for item in triad:
            if item.get("mono"):
                if not first_mono_found:
                    first_mono_found = True
                else:
                    item["mono"] = False
        return triad
    context["fallback_reason"] = "converted_to_mono"
    mono_candidates = [it for it in triad if CAT_BY_ID.get(it["id"], {}).get("mono")]
    if mono_candidates:
        mono_candidates.sort(key=lambda x: TIER_RANK.get(x.get("tier"), 0), reverse=True)
        mono_id_to_set = mono_candidates[0]["id"]
        for item in triad:
            item["mono"] = (item["id"] == mono_id_to_set)
    else:
        classic_item_found = False
        for item in triad:
            if item.get("tier") == "Classic":
                item["mono"] = True
                classic_item_found = True
                break
        if not classic_item_found:
            triad[0]["mono"] = True
    return triad

def _apply_edge_register_filters(pool: list[dict], edge_type: str) -> list[dict]:
    regs = EDGE_REGISTERS.get(edge_type) or {}
    if not regs: return pool
    lg_policy = regs.get("lg_policy", "allow")
    allow_lg_mix = bool(regs.get("allow_lg_in_mix", True))
    allow_lg_mono = bool(regs.get("allow_lg_in_mono", True))
    def _lg_ok(it, is_mono=False):
        if not it.get("luxury_grand"): return True
        if lg_policy == "block": return False
        return allow_lg_mono if is_mono else allow_lg_mix
    targets = set((regs.get("palette_targets") or []))
    avoid   = set((regs.get("palette_avoid") or []))
    prefer_species = set((regs.get("species_prefer") or []))
    avoid_species  = set((regs.get("species_avoid") or []))
    t_boost = float(regs.get("palette_target_boost", 1.0))
    a_pen   = float(regs.get("palette_avoid_penalty", 1.0))
    scored = []
    for it in pool:
        is_potential_mono = bool(it.get("mono"))
        if not _lg_ok(it, is_mono=is_potential_mono):
             continue
        score = 1.0
        pal = set(it.get("palette", []) or [])
        if pal & targets: score *= t_boost
        if pal & avoid:   score *= a_pen
        flw = set((it.get("flowers") or []))
        if flw & prefer_species: score *= 1.05
        if flw & avoid_species:  score *= 0.95
        scored.append((score, it))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in scored]

def _ensure_triad_or_500(items: list[dict]) -> None:
    triad_ok = isinstance(items, list) and len(items) == 3
    mono_count = sum(1 for it in items if it.get("mono") is True)
    mono_ok = (mono_count == 1)
    if triad_ok and mono_ok:
        return
    log.error("[TRIAD_CONTRACT] invalid triad: len=%s mono_count=%s ids=%s",
        (len(items) if isinstance(items, list) else None), mono_count,
        [_stable_id(it) for it in (items or [])])
    raise HTTPException(status_code=500, detail="Internal catalog data error")

def _compute_detected_emotions(scores: dict[str,float]) -> list[dict]:
    cfg = (ANCHOR_THRESHOLDS.get("multi_anchor_logging") or {})
    if not cfg.get("enabled", True): return []
    max_entries = int(cfg.get("max_entries", 2))
    score2_min = float(cfg.get("score2_min", 0.25))
    delta_max  = float(cfg.get("delta_max", 0.15))
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not ordered: return []
    top1 = ordered[0]
    out = [{"anchor": top1[0], "score": round(float(top1[1]), 2)}]
    if len(ordered) > 1:
        top2 = ordered[1]
        if float(top2[1]) >= score2_min and (float(top1[1]) - float(top2[1])) <= delta_max:
            out.append({"anchor": top2[0], "score": round(float(top2[1]), 2)})
    return out[:max_entries]

def detect_emotion(prompt: str, context: dict | None) -> Tuple[str, Optional[str], Dict[str, float]]:
    p = normalize(prompt)
    scores: Dict[str, float] = {}
    edge_type: Optional[str] = None
    for edge, rules in (EDGES or {}).items():
        if _match_edge(p, rules):
            edge_type = edge
            edge_config = EDGE_REGISTERS.get(edge) or {}
            resolved_anchor = edge_config.get("emotion_anchor") or "general"
            log.info(f"Edge case detected: {edge_type} -> {resolved_anchor}")
            return resolved_anchor, edge_type, scores
    tokens = _tokenize(prompt)
    buckets = EMOTION_KEYWORDS.get("keywords", {}) or {}
    for anchor, words in buckets.items():
        hits = sum(1 for w in (words or []) if (w or "").lower() in p)
        if hits > 0: scores[anchor] = float(hits)
    has_grand = _has_grand_intent(prompt)
    if has_grand:
        scores["luxury_grand"] = (scores.get("luxury_grand", 0.0) + 20.0) * 1.1
    log.debug(f"Detected scores: {scores}, grand_intent: {has_grand}")
    blocked = (TIER_POLICY.get("luxury_grand", {}) or {}).get("blocked_emotions") or []
    if has_grand and blocked:
        original_scores = dict(scores)
        scores = {k: v for k, v in scores.items() if k not in blocked}
        log.debug(f"Applied LG blocking for grand intent. Original: {original_scores}, After: {scores}")
    if not scores: return "general", None, {}
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    resolved_anchor = sorted_scores[0][0] if sorted_scores else "general"
    if resolved_anchor == "general" and len(sorted_scores) > 1:
        resolved_anchor = sorted_scores[1][0]
    return resolved_anchor, edge_type, scores

# --- B2 START: New function for relationship context detection ---
def detect_relationship_context(prompt: str) -> tuple[str, bool]:
    """
    Returns (context, ambiguous)
    context ∈ {"romantic","familial","friendship","professional"}
    """
    p = prompt.lower()
    romantic = {"love", "my love", "sweetheart", "darling", "partner"}
    familial = {"mom", "mother", "dad", "father", "sister", "brother", "family"}
    professional = {"team", "boss", "manager", "colleague", "work", "office"}
    friendship = {"friend", "buddy", "pal"}

    hits = []
    if any(w in p for w in romantic): hits.append("romantic")
    if any(w in p for w in familial): hits.append("familial")
    if any(w in p for w in professional): hits.append("professional")
    if any(w in p for w in friendship): hits.append("friendship")

    if not hits: return ("friendship", True)  # safest default
    if len(hits) == 1: return (hits[0], False)

    # tie-breaker (romantic > professional > familial > friendship)
    order = ["romantic", "professional", "familial", "friendship"]
    for k in order:
        if k in hits: 
            return (k, True)

    return (hits[0], True)
# --- B2 END ---

def find_iconic_species(prompt_norm: str) -> Optional[str]:
    if ONLY_ICONIC_RE.search(prompt_norm):
        for species, terms in ICONIC_SPECIES.items():
            if any(term in prompt_norm for term in terms):
                return species
    return None

def find_and_assign_note(triad: list, selected_species: Optional[str], selected_emotion: Optional[str], prompt_text: str = "") -> None:
    found_species = any(selected_species in it.get("flowers", []) for it in triad) if selected_species else False
    if selected_species and not found_species:
        note = SUB_NOTES.get("species_not_found", "We couldn't find a {species} bouquet...").replace("{species}", selected_species)
        target = next((it for it in triad if not it.get("mono")), triad[0] if triad else None)
        if target is not None: target["note"] = note

def _transform_for_api(items: List[Dict], resolved_anchor: Optional[str]) -> List[Dict]:
    _ensure_triad_or_500(items)
    out: List[Dict] = []
    for it in items:
        item = dict(it)
        item["image"] = item.get("image") or item.get("image_url") or "/static/default-flower.jpg"
        item["price"] = item.get("price") if item.get("price") is not None else item.get("price_inr", 1000)
        item["currency"] = "INR"
        item["emotion"] = item.get("emotion") or resolved_anchor or "general"
        if "image_url" in item: del item["image_url"]
        if "price_inr" in item: del item["price_inr"]
        out.append(item)
    return out

def _backfill_to_three(items: list[dict], catalog: list[dict], context: dict) -> list[dict]:
    if len(items) >= 3:
        return items[:3]
    final_triad = [None, None, None]
    existing_ids = set()
    for item in items:
        tier_idx = TIER_RANK.get(item.get("tier"))
        if tier_idx is not None:
            final_triad[tier_idx] = item
            existing_ids.add(_stable_id(item))
    missing_tiers = [TIER_ORDER[i] for i, item in enumerate(final_triad) if item is None]
    target_family = context.get("target_family")
    family_pool = [it for it in catalog if (it.get("sentiment_family") == target_family or it.get("sentiment_family") == "UNSPECIFIED") and _stable_id(it) not in existing_ids]
    for tier in missing_tiers:
        tier_pool = [it for it in family_pool if it.get("tier") == tier]
        if tier_pool:
            selected = tier_pool[0]
            final_triad[TIER_RANK[tier]] = selected
            existing_ids.add(_stable_id(selected))
            family_pool = [it for it in family_pool if _stable_id(it) != _stable_id(selected)]
    if None in final_triad and context.get("sentiment_over_ladder"):
        filled_items = [it for it in final_triad if it is not None]
        if filled_items:
            filler = filled_items[-1]
            for i in range(3):
                if final_triad[i] is None:
                    final_triad[i] = dict(filler)
                    final_triad[i]['tier'] = TIER_ORDER[i]
    if None in final_triad:
        context["fallback_reason"] = "cross_family_last_resort"
        any_pool = [it for it in catalog if _stable_id(it) not in existing_ids]
        for i in range(3):
            if final_triad[i] is None:
                tier_to_fill = TIER_ORDER[i]
                candidate = next((it for it in any_pool if it.get("tier") == tier_to_fill), None)
                if candidate:
                    final_triad[i] = candidate
                    existing_ids.add(_stable_id(candidate))
    final_triad = [it for it in final_triad if it is not None]
    while len(final_triad) < 3:
        final_triad.append(final_triad[-1] if final_triad else catalog[0])
    return final_triad[:3]

# --- B1 START: New helper functions for family boundaries ---
def _family_for_anchor(anchor: str, families_json: dict) -> str:
    families = families_json.get("sentiment_families", {})
    for fam, spec in families.items():
        emos = set(spec.get("emotions") or [])
        if anchor in emos:
            return fam
        # also allow nested subfamilies (reconciliation.*)
        if isinstance(spec, dict) and "subfamilies" in spec:
            for subname, sub in spec["subfamilies"].items():
                if isinstance(sub, dict) and anchor in set(sub.get("emotions") or []):
                    return subname # Return the specific subfamily name
    return "UNSPECIFIED"

def _filter_in_family(catalog: list[dict], fam: str, families_json: dict) -> list[dict]:
    families = families_json.get("sentiment_families", {})
    fam_emos = set()
    spec = families.get(fam)
    if not spec: # Handle subfamilies
        for main_fam in families.values():
            if isinstance(main_fam, dict) and "subfamilies" in main_fam and fam in main_fam["subfamilies"]:
                spec = main_fam["subfamilies"][fam]
                break
    
    if isinstance(spec, dict):
        fam_emos.update(spec.get("emotions") or [])

    return [x for x in catalog if (x.get("emotion") in fam_emos)]

def _pool_general_in_family(catalog: list[dict], fam: str) -> list[dict]:
    # convention: items with "emotion" == "general" but tagged with family in your data
    return [x for x in catalog if (x.get("emotion") == "general" and x.get("sentiment_family") == fam)]

def _fallback_boundary_path(available_catalog: list[dict], target_family: str, context: dict,
                            families_json: dict, need_count: int) -> list[dict]:
    # 1) in-family
    p1 = _filter_in_family(available_catalog, target_family, families_json)
    if len(p1) >= need_count:
        context["fallback_reason"] = "in_family"
        return p1

    # 2) general-in-family
    p2_candidates = _pool_general_in_family(available_catalog, target_family)
    p2 = p1 + [item for item in p2_candidates if item["id"] not in {x["id"] for x in p1}]
    if len(p2) >= need_count:
        context["fallback_reason"] = "general_in_family"
        return p2

    # 3) duplicate-tier (only if sentiment_over_ladder is true for this edge)
    # This path returns the current pool; the duplication logic is handled in the tier loop.
    if context.get("edge_type") in {"sympathy", "farewell"} and context.get("sentiment_over_ladder"):
        context["fallback_reason"] = "duplicate_tier"
        return p2

    # 4) last resort cross-family
    context["fallback_reason"] = "cross_family_last_resort"
    return available_catalog
# --- B1 END ---


def selection_engine(prompt: str, context: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    if not CATALOG:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    
    prompt_norm = normalize(prompt)
    resolved_anchor, edge_type, _scores = detect_emotion(prompt, context or {})
    selected_species = find_iconic_species(prompt_norm)
    
    # Ensure anchor/edge available to downstream
    context["resolved_anchor"] = resolved_anchor
    context["edge_type"] = edge_type
    
    # --- B2 START: Reroute apology lanes ---
    if edge_type == "apology":
        rel_ctx, ambiguous = detect_relationship_context(prompt)
        context["relationship_context"] = rel_ctx
        context["relationship_ambiguous"] = bool(ambiguous)
        # if romantic → prefer Affection/Support family; else Reconciliation lanes
        if rel_ctx == "romantic":
            resolved_anchor = "Affection/Support"  # keep evidence; family logic will constrain
    # --- B2 END ---
            
    # --- B1 START: Use new family boundary logic ---
    target_family = _family_for_anchor(resolved_anchor, SENTIMENT_FAMILIES)
    context["target_family"] = target_family
    sentiment_over_ladder = bool(EDGE_REGISTERS.get(edge_type, {}).get("sentiment_over_ladder", False))
    context["sentiment_over_ladder"] = sentiment_over_ladder
    
    available_catalog = list(CATALOG)
    # 'need_count' is how many mix candidates you require (usually 2 + slack)
    emotion_pool = _fallback_boundary_path(available_catalog, target_family, context, SENTIMENT_FAMILIES, need_count=6)
    # --- B1 END ---

    context["duplication_used"] = False
    context["fallback_reason"] = context.get("fallback_reason", "none") # Set by boundary path
    context["barriers_triggered"] = []
    
    seed = context.get("prompt_hash", "seed")
    triad = []
    seen_ids = set()

    for tier in TIER_ORDER:
        # The main pool is now pre-filtered by the boundary path logic
        tier_pool = [it for it in emotion_pool if it.get("tier") == tier and _stable_id(it) not in seen_ids]
        
        # We no longer need _filter_by_family or separate emotion matching here,
        # as it's handled by _fallback_boundary_path. We just pick from the result.
        
        if edge_type:
            tier_pool = _apply_edge_register_filters(tier_pool, edge_type)
        
        selected_item = None
        if tier_pool:
            idx = _rotation_index(f"{seed}:{tier}", len(tier_pool))
            selected_item = dict(tier_pool[idx])
        else:
            # Duplication logic is still valid if the boundary path allows it
            if context.get("fallback_reason") == "duplicate_tier" and triad:
                context["duplication_used"] = True
                if tier == "Signature" and triad[0]:
                    selected_item = dict(triad[0])
                elif tier == "Luxury" and len(triad) > 1 and triad[1]:
                     selected_item = dict(triad[1])
                elif triad[0]:
                    selected_item = dict(triad[0])
        
        if selected_item:
            selected_item["tier"] = tier
            triad.append(selected_item)
            seen_ids.add(_stable_id(selected_item))
        else:
            triad.append(None)
            
    if any(item is None for item in triad):
        triad_no_none = [item for item in triad if item is not None]
        final_triad = _backfill_to_three(triad_no_none, available_catalog, context)
    else:
        final_triad = triad[:3]

    pre_suppress_pool_size = {t: len([it for it in available_catalog if it.get("tier") == t]) for t in TIER_ORDER}
    if context.get("recent_ids"):
        final_triad = _suppress_recent(final_triad, set(context["recent_ids"]))
        if len(final_triad) < 3:
            final_triad = _backfill_to_three(final_triad, CATALOG, context)
            
    post_suppress_pool_size = {t: sum(1 for it in final_triad if it.get("tier") == t) for t in TIER_ORDER}
    final_triad = _ensure_one_mono_in_triad(final_triad, context)
    _ensure_triad_or_500(final_triad)
    
    if edge_type:
        for it in final_triad:
            it["edge_case"] = True
            it["edge_type"] = edge_type
            
    find_and_assign_note(final_triad, selected_species, resolved_anchor, prompt)
    if not _intent_clarity(prompt, len(_scores)):
        _add_unclear_mix_note(final_triad)
    
    context["pool_sizes"] = {
        "pre_suppress": pre_suppress_pool_size,
        "post_suppress": post_suppress_pool_size
    }
    
    meta_detected = []
    if FEATURE_MULTI_ANCHOR_LOGGING:
        meta_detected = _compute_detected_emotions(_scores or {})
    
    log_record = {
        "resolved_anchor": context.get("resolved_anchor"),
        "edge_case": bool(context.get("edge_type")),
        "edge_type": context.get("edge_type"),
        "pool_sizes": context.get("pool_sizes"),
        "fallback_reason": context.get("fallback_reason", "none"),
        "duplication_used": context.get("duplication_used", False),
        "sentiment_override_used": context.get("sentiment_over_ladder", False)
    }
    log.info("SELECTION_EVIDENCE: %s", json.dumps(log_record, ensure_ascii=False))

    for c in final_triad:
        c["emotion"] = c.get("emotion") or context.get("resolved_anchor") or "general"
        
    meta = {"detected_emotions": meta_detected}

    return final_triad, context, meta
