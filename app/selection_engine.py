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

# --- Drift-Proofing: Enumerate fallback reasons in one place ---
FALLBACK_REASONS = frozenset({"in_family", "general_in_family", "duplicate_tier", "cross_family_last_resort"})

def _rel_terms(key: str, fallback: set[str]) -> set[str]:
    src = (EMOTION_KEYWORDS.get("relationship_signals") or {}).get(key) or []
    # accept both lists and dict-likes; coerce to lowercase strings
    out = { (str(t or "")).strip().lower() for t in src if str(t or "").strip() }
    return out or { s.lower() for s in fallback }

# Relationship context tokens (from emotion_keywords.json with fallback)
ROMANTIC_TOKENS    = _rel_terms("romantic",    {
    "love","romance","romantic","blush","sweetheart","darling","my love","crush",
    "miss you","thinking of you","together","couple","partner","wife","husband",
    "girlfriend","boyfriend","fiancé","anniversary","hearts","date night","soulmate",
    "adoration","forever","valentine","my valentine","sweetheart"
})
PROFESSIONAL_TOKENS = _rel_terms("professional", {
    "colleague","coworker","teammate","work","office","team","project","deadline",
    "years of service","milestone","new role","promotion","manager","client"
})
FAMILIAL_TOKENS     = _rel_terms("familial",    {
    "mother","mom","mum","father","dad","parent","parents","son","daughter","brother",
    "sister","grandma","grandmother","granny","grandpa","grandfather","aunt","uncle",
    "cousin","niece","nephew","in-law","mother-in-law","father-in-law","family","familial"
})
FRIENDSHIP_TOKENS   = _rel_terms("friendship",  {
    "friend","friends","best friend","bestie","bff","buddy","pal","mate","bros","homie","friendship"
})

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

def detect_relationship_context(prompt_norm: str) -> tuple[str, bool]:
    p = prompt_norm
    hits = {
        "romantic": any(tok in p for tok in ROMANTIC_TOKENS),
        "familial": any(tok in p for tok in FAMILIAL_TOKENS),
        "friendship": any(tok in p for tok in FRIENDSHIP_TOKENS),
        "professional": any(tok in p for tok in PROFESSIONAL_TOKENS),
    }
    positives = [k for k,v in hits.items() if v]
    if not positives:
        return "unknown", True
    # priority: romantic > professional > familial > friendship
    for lane in ("romantic","professional","familial","friendship"):
        if lane in positives:
            return lane, (len(positives) > 1)
    return "unknown", True

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
        if tier_idx is not None and final_triad[tier_idx] is None:
            final_triad[tier_idx] = item
            existing_ids.add(_stable_id(item))
            
    target_family = context.get("target_family")
    family_pool = [it for it in catalog if (it.get("sentiment_family") == target_family or it.get("sentiment_family") == "UNSPECIFIED") and _stable_id(it) not in existing_ids]

    if context.get("edge_type"):
        family_pool = _apply_edge_register_filters(family_pool, context["edge_type"])
        
    missing_tiers = [TIER_ORDER[i] for i, item in enumerate(final_triad) if item is None]

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

def _family_for_anchor(anchor: str, families_json: dict) -> str:
    families = families_json.get("sentiment_families", {})
    for fam, spec in families.items():
        emos = set(spec.get("emotions") or [])
        if anchor in emos:
            return fam
        if isinstance(spec, dict) and "subfamilies" in spec:
            for subname, sub in spec["subfamilies"].items():
                if isinstance(sub, dict) and anchor in set(sub.get("emotions") or []):
                    return subname
    return "UNSPECIFIED"

def _filter_in_family(catalog: list[dict], fam: str, families_json: dict) -> list[dict]:
    families = families_json.get("sentiment_families", {})
    fam_emos = set()
    spec = families.get(fam)
    if not spec:
        for main_fam in families.values():
            if isinstance(main_fam, dict) and "subfamilies" in main_fam and fam in main_fam["subfamilies"]:
                spec = main_fam["subfamilies"][fam]
                break
    
    if isinstance(spec, dict):
        fam_emos.update(spec.get("emotions") or [])

    return [x for x in catalog if (x.get("emotion") in fam_emos)]

def _pool_general_in_family(catalog: list[dict], fam: str) -> list[dict]:
    return [x for x in catalog if (x.get("emotion") == "general" and x.get("sentiment_family") == fam)]

def _fallback_boundary_path(available_catalog: list[dict], target_family: str, context: dict,
                            families_json: dict, need_count: int) -> list[dict]:
    if not context.get("fallback_reason"):
        context["fallback_reason"] = "none"

    # Build a barrier-aware base pool once
    base_pool = _filter_by_family(available_catalog, target_family, context)
    
    # 1) in-family (emotion-specific)
    p1 = [x for x in base_pool if x.get("emotion") in set((_get_family_config(target_family) or {}).get("emotions") or [])]
    if len(p1) >= need_count:
        context["fallback_reason"] = "in_family"
        return p1

    # 2) general-in-family
    p2_candidates = [x for x in base_pool if x.get("emotion") == "general"]
    p2 = p1 + [item for item in p2_candidates if item["id"] not in {x["id"] for x in p1}]
    if len(p2) >= need_count:
        context["fallback_reason"] = "general_in_family"
        return p2

    # 3) duplicate-tier
    if context.get("sentiment_over_ladder"):
        context["fallback_reason"] = "duplicate_tier"
        return p2

    # 4) last resort cross-family
    context["fallback_reason"] = "cross_family_last_resort"
    return available_catalog

def selection_engine(prompt: str, context: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    if not CATALOG:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    
    prompt_norm = normalize(prompt)
    resolved_anchor, edge_type, _scores = detect_emotion(prompt, context or {})
    selected_species = find_iconic_species(prompt_norm)
    
    context["resolved_anchor"] = resolved_anchor
    context["edge_type"] = edge_type
    
    ctx, amb = detect_relationship_context(prompt_norm)
    context["relationship_context"] = ctx
    context["relationship_ambiguous"] = amb

    if context.get("edge_type") == "apology":
        rc = context.get("relationship_context")
        if rc == "romantic":
            context["sentiment_family"] = "romantic_repair"
        elif rc in {"familial","friendship","professional"}:
            context["sentiment_family"] = f"{rc}_repair"
        else:
            context["sentiment_family"] = "friendship_repair"
            
    target_family = context.get("sentiment_family") or _family_for_anchor(resolved_anchor, SENTIMENT_FAMILIES)
    context["target_family"] = target_family
    sentiment_over_ladder = bool(EDGE_REGISTERS.get(edge_type, {}).get("sentiment_over_ladder", False))
    context["sentiment_over_ladder"] = sentiment_over_ladder
    
    available_catalog = list(CATALOG)
    emotion_pool = _fallback_boundary_path(available_catalog, target_family, context, SENTIMENT_FAMILIES, need_count=6)

    # B3: Pool-size telemetry (pre-suppression)
    pool_sizes = {
        "pre_suppress": {
            "classic": sum(1 for x in emotion_pool if x.get("tier")=="Classic"),
            "signature": sum(1 for x in emotion_pool if x.get("tier")=="Signature"),
            "luxury": sum(1 for x in emotion_pool if x.get("tier")=="Luxury"),
        }
    }

    context["duplication_used"] = False
    context["barriers_triggered"] = []
    
    seed = context.get("prompt_hash", "seed")
    triad = []
    seen_ids = set()

    for tier in TIER_ORDER:
        tier_pool = [it for it in emotion_pool if it.get("tier") == tier and _stable_id(it) not in seen_ids]
        
        if edge_type:
            tier_pool = _apply_edge_register_filters(tier_pool, edge_type)
        
        selected_item = None
        if tier_pool:
            idx = _rotation_index(f"{seed}:{tier}", len(tier_pool))
            selected_item = dict(tier_pool[idx])
        else:
            if context.get("fallback_reason") == "duplicate_tier" and triad:
                context["duplication_used"] = True
                if tier == "Signature" and triad[0]: selected_item = dict(triad[0])
                elif tier == "Luxury" and len(triad) > 1 and triad[1]: selected_item = dict(triad[1])
                elif triad[0]: selected_item = dict(triad[0])
        
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

    # compute a post-suppress view of the pool (exclude recent_ids)
    recent_set = set(context.get("recent_ids") or [])
    post_pool = [x for x in emotion_pool if _stable_id(x) not in recent_set]

    pool_sizes["post_suppress"] = {
        "classic":   sum(1 for x in post_pool if x.get("tier") == "Classic"),
        "signature": sum(1 for x in post_pool if x.get("tier") == "Signature"),
        "luxury":    sum(1 for x in post_pool if x.get("tier") == "Luxury"),
    }
            
    final_triad = _ensure_one_mono_in_triad(final_triad, context)
    _ensure_triad_or_500(final_triad)
    
    if edge_type:
        for it in final_triad:
            it["edge_case"] = True
            it["edge_type"] = edge_type
            
    find_and_assign_note(final_triad, selected_species, resolved_anchor, prompt)
    if not _intent_clarity(prompt, len(_scores)):
        _add_unclear_mix_note(final_triad)
    
    meta_detected = []
    if FEATURE_MULTI_ANCHOR_LOGGING:
        meta_detected = _compute_detected_emotions(_scores or {})
    
    log_record = {
        "resolved_anchor": context.get("resolved_anchor"),
        "edge_case": bool(context.get("edge_type")),
        "edge_type": context.get("edge_type"),
        "fallback_reason": context.get("fallback_reason", "none"),
        "duplication_used": context.get("duplication_used", False),
        "sentiment_override_used": context.get("sentiment_over_ladder", False),
        "relationship_context": context.get("relationship_context"),
        "relationship_ambiguous": bool(context.get("relationship_ambiguous")),
        "pool_size": pool_sizes
    }
    log.info("SELECTION_EVIDENCE: %s", json.dumps(log_record, ensure_ascii=False))

    for c in final_triad:
        c["emotion"] = c.get("emotion") or context.get("resolved_anchor") or "general"
        
    meta = {"detected_emotions": meta_detected}

    return final_triad, context, meta

}

{
type: uploaded file
fileName: test_family_boundaries.py
fullContent:
from app.selection_engine import selection_engine
import pytest

# B1: Family boundaries ON by default
# Sympathy/farewell must not leak Celebration palettes.

PROMPTS_AND_FORBIDDEN_PALETTES = [
    ("i'm so sorry for your loss", {"deep-red", "crimson", "gold", "bright", "hot-pink"}),
    ("condolences on your loss", {"deep-red", "crimson", "gold", "bright", "hot-pink"}),
    ("farewell flowers", {"deep-red", "crimson", "gold", "bright", "hot-pink"}),
    ("farewell to a colleague", {"deep-red", "crimson", "gold", "bright", "hot-pink"}),
]

@pytest.mark.parametrize("prompt,forbidden", PROMPTS_AND_FORBIDDEN_PALETTES)
def test_sympathy_farewell_palette_guard(prompt, forbidden):
    """Verifies that sympathy/farewell prompts do not return items with celebration palettes."""
    items, _, _ = selection_engine(prompt=prompt, context={})
    assert len(items) == 3, "Expected a triad of 3 items"
    
    for item in items:
        palette = set(item.get("palette", []))
        assert not (palette & forbidden), \
            f"Item {item.get('id')} for prompt '{prompt}' has forbidden palette token. "\
            f"Found: {palette & forbidden}"

}

{
type: uploaded file
fileName: main.py
fullContent:
# main.py
import os, json, logging, uuid, time, csv
from datetime import datetime
from typing import Any, Dict, List, Optional

# --- P1.4a Nit: Import Response for header injection ---
from fastapi import FastAPI, HTTPException, Request, Response, Body
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

from .selection_engine import selection_engine, normalize, detect_emotion, _transform_for_api

# =========================
# Environment & Constants
# =========================
ALLOWED = os.getenv("ALLOWED_ORIGINS", "https://arvyam.com")
ASSET_BASE = os.getenv("PUBLIC_ASSET_BASE", "https://arvyam.com").rstrip("/")
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "10"))
PERSONA = os.getenv("PERSONA_NAME", "ARVY")  # for logs/UI
ERROR_PERSONA = "ARVY"                       # hard-coded in API errors
ICONIC = {"rose","lily","orchid"}            # for analytics guard
ANALYTICS_ENABLED = os.getenv("ANALYTICS_ENABLED", "0") == "1"
GOLDEN_ARTIFACTS_DIR = os.getenv("GOLDEN_ARTIFACTS_DIR", "/tmp/arvy_golden")


# =========================
# Logging
# =========================
# --- P1.4a Nit: Standardize on the named logger instance ---
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

def load_json(relpath: str, default=None):
    try:
        with open(os.path.join(HERE, relpath), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

CATALOG: List[Dict[str, Any]] = load_json("catalog.json", default=[])
CAT_BY_ID: Dict[str, Dict[str, Any]] = {it["id"]: it for it in CATALOG if isinstance(it, dict) and "id" in it}

def _load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

ROOT = os.path.dirname(__file__)
RULES_DIR = os.path.join(ROOT, "rules")
ANCHOR_THRESHOLDS = _load_json(os.path.join(RULES_DIR, "anchor_thresholds.json"), {})
FEATURE_MULTI_ANCHOR_LOGGING = os.getenv("FEATURE_MULTI_ANCHOR_LOGGING", "0") == "1"

# =========================
# Seed rollback utilities
# =========================
SEED_FILE = os.path.join(HERE, "rules", "seed_triads.json")
SEED_TOGGLE_FILE = os.path.join(HERE, "rules", "seed_toggle.json")

def load_seeds() -> Dict[str, List[str]]:
    seeds = load_json(os.path.relpath(SEED_FILE, HERE), default={})
    return seeds or {}

def _now() -> int:
    return int(time.time())

def seed_mode_status() -> Dict[str, Any]:
    data = load_json(os.path.relpath(SEED_TOGGLE_FILE, HERE), default={"enabled": False, "until": 0})
    # auto-expire
    if data.get("enabled") and _now() >= int(data.get("until", 0)):
        data = {"enabled": False, "until": 0}
        try:
            with open(SEED_TOGGLE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass
    return data

def enable_seed_mode(minutes: int = 60) -> Dict[str, Any]:
    data = {"enabled": True, "until": _now() + max(1, int(minutes))*60}
    os.makedirs(os.path.dirname(SEED_TOGGLE_FILE), exist_ok=True)
    with open(SEED_TOGGLE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data

def disable_seed_mode() -> Dict[str, Any]:
    data = {"enabled": False, "until": 0}
    os.makedirs(os.path.dirname(SEED_TOGGLE_FILE), exist_ok=True)
    with open(SEED_TOGGLE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data

def map_ids_to_output(ids: List[str]) -> Optional[List[Dict[str, Any]]]:
    """Builds output triad from catalog ids. Returns None if invalid/missing data."""
    out: List[Dict[str, Any]] = []
    for pid in ids:
        it = CAT_BY_ID.get(pid)
        if not it: return None
        pal = it.get("palette") or []
        if not isinstance(pal, list) or not pal: return None
        out.append({
            "id": it["id"],
            "title": it["title"],
            "desc": it["desc"],
            "image": it["image_url"],
            "price": it["price_inr"],
            "currency": "INR",
            "emotion": it["emotion"],
            "tier": it["tier"],
            "packaging": it.get("packaging"),
            "mono": bool(it.get("mono")),
            "palette": pal,
            "luxury_grand": bool(it.get("luxury_grand"))
        })
    # Validate 2 MIX + 1 MONO
    if len(out) != 3: return None
    if sum(1 for x in out if x["mono"]) != 1: return None
    ids_set = set()
    for x in out:
        if x["id"] in ids_set: return None
        ids_set.add(x["id"])
    return out

# =========================
# Helpers
# =========================
def sanitize_text(s: str) -> str:
    return " ".join((s or "").strip().split())

def write_golden_artifact(
    request_id: str,
    persona: str,
    context: Dict[str, Any],
    items: List[Dict[str, Any]]
) -> None:
    """Writes a single-line JSON artifact to a date-stamped log file."""
    try:
        log_dir = GOLDEN_ARTIFACTS_DIR
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{datetime.utcnow().strftime('%Y%m%d')}.log")
        artifact = {
            "ts": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "persona": persona,
            "resolved_anchor": context.get("resolved_anchor"),
            "item_ids": [item.get("id") for item in items],
            "fallback_reason": context.get("fallback_reason"),
            "pool_sizes": context.get("pool_sizes"),
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(artifact) + "\n")
    except Exception as e:
        logger.error(f"Failed to write golden artifact for request_id={request_id}: {e}")

def append_selection_log(items: List[Dict[str, Any]], request_id: str, latency_ms: int, prompt_len: int, path: str = "/api/curate") -> None:
    """Appends one analytics row per curate request."""
    logs_dir = os.path.join(HERE, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    csv_path = os.path.join(logs_dir, "selection_log.csv")
    header = ["ts","request_id","persona","path","latency_ms","prompt_len","detected_emotion","mix_ids","mono_id","tiers","luxury_grand_flags"]
    detected_emotion = items[0].get("emotion") if items else ""
    mix_ids = ";".join([it["id"] for it in items if not it.get("mono")])
    mono_id = next((it["id"] for it in items if it.get("mono")), "")
    tiers = ";".join([it.get("tier","") for it in items])
    lg_flags = ";".join(["true" if it.get("luxury_grand") else "false" for it in items])
    row = [time.strftime("%Y-%m-%dT%H:%M:%S%z"), request_id, PERSONA, path, str(latency_ms), str(prompt_len), detected_emotion, mix_ids, mono_id, tiers, lg_flags]
    need_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(header)
        w.writerow(row)

def analytics_guard_check() -> Dict[str, Any]:
    """Read last 50 rows of selection_log.csv and compute R/L/O share in MIX items."""
    if not ANALYTICS_ENABLED:
        return {"window": 0, "mix_iconic_ratio": None, "alert": False, "message": "Analytics disabled."}
    logs_dir = os.path.join(HERE, "logs")
    csv_path = os.path.join(logs_dir, "selection_log.csv")
    result = {"window": 0, "mix_iconic_ratio": None, "alert": False, "message": ""}
    if not os.path.exists(csv_path): return result
    with open(csv_path, "r", encoding="utf-8") as f: rows = list(csv.reader(f))
    if len(rows) <= 1: return result
    data, total_mix, iconic_mix = rows[1:][-50:], 0, 0
    for r in data:
        try:
            mix_ids = [x for x in (r[7] if len(r) > 7 else "").split(";") if x]
            for mid in mix_ids:
                total_mix += 1
                it = CAT_BY_ID.get(mid)
                if it and any((f.lower() in ICONIC) for f in (it.get("flowers") or [])): iconic_mix += 1
        except Exception: continue
    result["window"] = len(data)
    if total_mix > 0:
        ratio = iconic_mix / float(total_mix)
        result["mix_iconic_ratio"] = ratio
        if ratio < 0.5:
            result["alert"] = True
            result["message"] = f"Iconic MIX share dipped below 50% over the last {len(data)} requests."
    os.makedirs(logs_dir, exist_ok=True)
    with open(os.path.join(logs_dir, "analytics_guard.json"), "w", encoding="utf-8") as f: json.dump(result, f, indent=2)
    if result["alert"]: logger.warning("[GUARD] MIX iconic ratio %.2f < 0.5 over last %s requests", result.get("mix_iconic_ratio", 0.0), result["window"])
    else: logger.info("[GUARD] MIX iconic ratio %s over last %s requests", f"{result.get('mix_iconic_ratio', 0.0):.2f}" if result.get("mix_iconic_ratio") is not None else "n/a", result["window"])
    return result

async def _coerce_prompt(request: Request) -> str:
    try: data = await request.json()
    except Exception: data = None
    if isinstance(data, dict):
        for k in ("prompt", "text", "q", "message", "input"):
            if isinstance(v := data.get(k), str) and v.strip(): return v.strip()
        for v in data.values():
            if isinstance(v, str) and v.strip(): return v.strip()
    try: form = await request.form()
    except Exception: form = None
    if form:
        for k in ("prompt", "text", "q", "message"):
            if k in form and str(form[k]).strip(): return str(form[k]).strip()
    raw = (await request.body() or b"").decode("utf-8", "ignore").strip()
    if raw and not (raw.startswith("{") or raw.startswith("[")): return raw
    for k in ("prompt", "text", "q", "message", "input"):
        if (v := request.query_params.get(k)) and v.strip(): return v.strip()
    return ""

# =========================
# Schemas (UI-aligned)
# =========================
class CurateContext(BaseModel):
    budget_inr: Optional[int] = None
    emotion_hint: Optional[str] = None
    packaging_pref: Optional[str] = None
    locale: Optional[str] = None
    recent_ids: Optional[List[str]] = None
    session_id: Optional[str] = None
    run_count: Optional[int] = 0

class CurateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500)
    context: Optional[CurateContext] = None

class SeedModeIn(BaseModel):
    minutes: Optional[int] = 60

class CheckoutIn(BaseModel):
    product_id: str

class ItemOut(BaseModel):
    id: str
    title: str
    desc: str
    image: str
    price: int
    currency: str
    emotion: str
    tier: str
    packaging: Optional[str] = None
    mono: bool
    palette: List[str]
    luxury_grand: bool
    edge_case: Optional[bool] = False
    edge_type: Optional[str] = None
    note: Optional[str] = None

# =========================
# Routes
# =========================
@app.get("/health")
def health(): return {"status": "ok", "persona": ERROR_PERSONA, "version": "v1"}

@app.get("/api/curate/seed_mode")
def seed_status(): return seed_mode_status()

@app.post("/api/curate/seed_mode")
def seed_enable(body: SeedModeIn): return enable_seed_mode(max(1, int(body.minutes or 60)))

@app.post("/api/curate/seed_mode/disable")
def seed_disable(): return disable_seed_mode()

@app.post("/api/curate", summary="Curate", response_model=List[ItemOut])
# --- P1.4a Nit: Add rate limiting to the primary typed endpoint ---
@limiter.limit(f"{RATE_LIMIT_PER_MIN}/minute")
async def curate_post(body: CurateRequest, request: Request, response: Response):
    """JSON-only canonical endpoint. Returns items validated against the ItemOut schema."""
    started = time.time()
    request_id = str(uuid.uuid4())
    prompt = body.prompt.strip()
    req_context = body.context.dict() if isinstance(body.context, CurateContext) else {}
    req_context["request_id"] = request_id

    try:
        final_triad, context, meta = selection_engine(prompt=prompt, context=req_context)
        items = _transform_for_api(final_triad, context.get("resolved_anchor"))
    except Exception as e:
        logger.error(f"Engine error for request_id={request_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": {"code": "ENGINE_ERROR", "message": str(e)[:400]}})

    latency_ms = int((time.time() - started) * 1000)
    logger.info("[%s] CURATE ip=%s emotion=%s latency_ms=%s prompt_len=%s",
                 PERSONA, get_remote_address(request), items[0].get("emotion",""), latency_ms, len(prompt))
    
    append_selection_log(items, request_id, latency_ms, len(prompt), path="/api/curate")
    analytics_guard_check()
    write_golden_artifact(request_id, PERSONA, context, items)

    emit_header = ANCHOR_THRESHOLDS.get("multi_anchor_logging", {}).get("emit_header", False)
    if FEATURE_MULTI_ANCHOR_LOGGING and emit_header:
        _, _, scores = detect_emotion(normalize(prompt), context or {})
        if scores:
            top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            s1 = round(top[0][1], 2)
            header_vals = [f"{top[0][0]}:{s1}"]
            if len(top) > 1:
                th = ANCHOR_THRESHOLDS.get("multi_anchor_logging", {})
                s2 = round(top[1][1], 2)
                if s2 >= float(th.get("score2_min", 0.25)) and (s1 - s2) <= float(th.get("delta_max", 0.15)):
                    header_vals.append(f"{top[1][0]}:{s2}")
            # --- P1.4a Nit: Set header on injected response object ---
            response.headers["X-Detected-Emotions"] = ",".join(header_vals)

    # --- P1.4a Nit: Return raw list to allow Pydantic validation via response_model ---
    return items

# =========================
# Flexible/Legacy Routes
# =========================
async def _curate_flexible(request: Request) -> JSONResponse:
    """Coercion shim for legacy/alias routes."""
    started = time.time()
    request_id = str(uuid.uuid4())
    prompt = await _coerce_prompt(request)
    if not prompt:
        return error_json("PROMPT_REQUIRED", "Provide a non-empty 'prompt'.", 400, request_id)
    
    prompt = sanitize_text(prompt)
    req_context = {"request_id": request_id}

    try:
        final_triad, context, meta = selection_engine(prompt=prompt, context=req_context)
        items = _transform_for_api(final_triad, context.get("resolved_anchor"))
    except Exception as e:
        logger.error(f"Engine error for request_id={request_id} (shim): {e}", exc_info=True)
        return error_json("ENGINE_ERROR", str(e)[:400], 500, request_id)

    latency_ms = int((time.time() - started) * 1000)
    logger.info("[%s] CURATE(SHIM) ip=%s emotion=%s latency_ms=%s prompt_len=%s",
                 PERSONA, get_remote_address(request), items[0].get("emotion",""), latency_ms, len(prompt))
    append_selection_log(items, request_id, latency_ms, len(prompt), path="/curate(shim)")
    analytics_guard_check()
    write_golden_artifact(request_id, PERSONA, context, items)
    
    return JSONResponse(items)

@app.get("/api/curate")
@limiter.limit(f"{RATE_LIMIT_PER_MIN}/minute")
async def curate_get(request: Request):
    # This endpoint retains its original seed-mode logic, which was previously removed by mistake.
    started = time.time()
    request_id = str(uuid.uuid4())
    prompt = await _coerce_prompt(request)
    if not isinstance(prompt, str) or len(prompt.strip()) < 3:
        return JSONResponse({"error": "Invalid input."}, status_code=400)
    
    prompt = sanitize_text(prompt)
    if not prompt:
        return error_json("EMPTY_PROMPT", "Please write a short line.", 422, request_id)
    
    req_context = {"request_id": request_id}
    
    seed_state = seed_mode_status()
    items = None
    context = {}
    if seed_state.get("enabled"):
        norm = normalize(prompt)
        emo, _, _ = detect_emotion(norm, req_context)
        seeds = load_seeds().get(emo) or []
        if len(seeds) == 3:
            seeded_items = map_ids_to_output(seeds)
            if seeded_items:
                # In seed mode, context is minimal. We pass the detected emotion as the resolved anchor.
                items = _transform_for_api(seeded_items, emo)
                context = {"resolved_anchor": emo} # Create a minimal context for logging

    if items is None:
        final_triad, context, meta = selection_engine(prompt=prompt, context=req_context)
        items = _transform_for_api(final_triad, context.get("resolved_anchor"))

    latency_ms = int((time.time() - started) * 1000)
    logger.info("[%s] CURATE ip=%s emotion=%s latency_ms=%s prompt_len=%s%s",
                 PERSONA, get_remote_address(request), items[0].get("emotion",""), latency_ms, len(prompt),
                 " (seed-mode)" if seed_state.get("enabled") and items is not None else "")
    
    append_selection_log(items, request_id, latency_ms, len(prompt), path="/api/curate_get")
    analytics_guard_check()
    write_golden_artifact(request_id, PERSONA, context, items)

    return JSONResponse(items)

@app.post("/curate")
@limiter.limit(f"{RATE_LIMIT_PER_MIN}/minute")
async def curate_alias_post(request: Request): return await _curate_flexible(request)

@app.get("/curate")
@limiter.limit(f"{RATE_LIMIT_PER_MIN}/minute")
async def curate_alias_get(request: Request): return await curate_get(request)

@app.post("/api/curate/next", summary="Curate (rotate next)", response_model=List[ItemOut])
async def curate_post_next(body: CurateRequest, request: Request):
    """Gets the next set of items, ensuring the response shape is identical to the primary endpoint."""
    request_id, req_context = str(uuid.uuid4()), body.context.dict() if isinstance(body.context, CurateContext) else {}
    req_context["run_count"] = int(req_context.get("run_count") or 0) + 1
    req_context["request_id"] = request_id
    try:
        final_triad, context, meta = selection_engine(prompt=body.prompt.strip(), context=req_context)
        items = _transform_for_api(final_triad, context.get("resolved_anchor"))
    except Exception as e:
        logger.error(f"Engine error for request_id={request_id} (next): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": {"code": "ENGINE_ERROR", "message": str(e)[:400]}})
    return items

# -------------------------
# Golden-Set Harness (Internal Test Tool)
# -------------------------
GOLDEN_TESTS: List[Dict[str, Any]] = [
    {"name": "romance_budget_2000", "prompt": "romantic anniversary under 2000", "context": {"budget_inr": 2000}},
    {"name": "romance_plain", "prompt": "romantic bouquet please"},
    {"name": "only_lilies", "prompt": "only lilies please", "expect_lily_mono": True},
    {"name": "hydrangea_redirect", "prompt": "hydrangea bouquet", "expect_note": True},
    {"name": "celebration_bright", "prompt": "bright congratulations"},
    {"name": "encouragement_exams", "prompt": "encouragement for exams"},
    {"name": "gratitude_thanks", "prompt": "thank you flowers"},
    {"name": "friendship_care", "prompt": "for a dear friend"},
    {"name": "encouragement_getwell", "prompt": "get well soon flowers"},
    {"name": "birthday", "prompt": "birthday flowers"},
    {"name": "sympathy_loss", "prompt": "i’m so sorry for your loss"},
    {"name": "apology", "prompt": "i deeply apologize"},
    {"name": "farewell", "prompt": "farewell flowers"},
    {"name": "valentine", "prompt": "valentine surprise"}
]

def _run_one(test: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"name": test["name"], "prompt": test["prompt"], "status": "PASS", "reasons": []}
    try: items, _, _ = selection_engine(prompt=test["prompt"], context=test.get("context") or {})
    except Exception as e: out["status"] = "FAIL"; out["reasons"].append(f"engine_error: {repr(e)}"); return out
    if len(items) != 3: out["status"] = "FAIL"; out["reasons"].append("triad_len != 3")
    if sum(1 for it in items if it.get("mono")) != 1: out["status"] = "FAIL"; out["reasons"].append("mono_count != 1")
    if any(not isinstance(it.get("palette"), list) or len(it["palette"]) == 0 for it in items): out["status"] = "FAIL"; out["reasons"].append("palette missing")
    if sum(1 for it in items if it.get("luxury_grand")) > 1: out["status"] = "FAIL"; out["reasons"].append(">1 luxury_grand")
    if test.get("expect_lily_mono"):
        mono_item = next((it for it in items if it.get("mono")), None)
        if not mono_item: out["status"] = "FAIL"; out["reasons"].append("no mono item")
        else:
            cat = CAT_BY_ID.get(mono_item["id"]) or {}
            if "lily" not in [s.lower() for s in (cat.get("flowers") or [])]: out["status"] = "FAIL"; out["reasons"].append("mono is not lily")
    if test.get("expect_note") and not any(it.get("note") for it in items): out["status"] = "FAIL"; out["reasons"].append("redirection note missing")
    out["emotion"] = items[0].get("emotion", ""); out["ids"] = [it.get("id") for it in items]
    out["tiers"] = [it.get("tier") for it in items]; out["mono_id"] = next((it.get("id") for it in items if it.get("mono")), "")
    return out

@app.post("/api/curate/golden")
def curate_golden(request: Request):
    started = time.time()
    results = [_run_one(t) for t in GOLDEN_TESTS]
    passed = sum(1 for r in results if r["status"] == "PASS")
    summary = {
        "ts": datetime.utcnow().isoformat(), "persona": ERROR_PERSONA, "total": len(results),
        "passed": passed, "failed": len(results) - passed,
        "latency_ms": int((time.time() - started) * 1000), "results": results
    }
    # --- P1.4a Nit: Write a small artifact per run for easy diffing ---
    try:
        logs_dir = os.path.join(HERE, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        with open(os.path.join(logs_dir, "golden_harness_runs.log"), "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
    except Exception as e:
        logger.error(f"Failed to write golden harness artifact: {e}")
    return summary

@app.get("/api/curate/golden/p1")
def curate_golden_p1(request: Request):
    """Alias for Phase-1 golden suite."""
    return curate_golden(request)

@app.post("/api/checkout")
@limiter.limit(f"{RATE_LIMIT_PER_MIN}/minute")
def checkout(body: CheckoutIn, request: Request):
    pid = (body.product_id or "").strip()
    if not pid: return error_json("BAD_PRODUCT", "product_id is required.", 422)
    url = f"https://checkout.example/intent?pid={pid}"
    logger.info(f"[{PERSONA}] CHECKOUT ip={get_remote_address(request)} product={pid}")
    return {"checkout_url": url}

# =========================
# Error Normalization
# =========================
def error_json(code: str, message: str, status: int = 400, request_id: Optional[str] = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}, "persona": ERROR_PERSONA, "request_id": request_id or str(uuid.uuid4())})

@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict):
        shaped = {"error": exc.detail.get("error", {"code": "HTTP_ERROR", "message": "Request error."}), "persona": ERROR_PERSONA, "request_id": str(uuid.uuid4())}
        return JSONResponse(status_code=exc.status_code, content=shaped)
    return await http_exception_handler(request, exc)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc): return error_json("VALIDATION_ERROR", "Invalid input.", 422)

@app.exception_handler(RateLimitExceeded)
async def ratelimit_handler(request: Request, exc: RateLimitExceeded): return error_json("RATE_LIMITED", "Too many requests. Please try again in a minute.", 429)
