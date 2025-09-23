#!/usr/bin/env python3
# tools/mine_unknowns.py
#
# Mine "unknown" user phrases from SELECTION_EVIDENCE logs for the offline feeder.
# OFFLINE-ONLY. Never runs in the request path.

import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set


# ------------------------------
# Normalization & helpers
# ------------------------------

WORD_RE = re.compile(r"[a-z0-9]+")

# Small English stopword list (safe: does NOT include domain words like "sorry")
DEFAULT_STOPWORDS = {
    "a","an","the","and","or","but","if","so","of","for","to","in","on","at","with","from","by","about","as",
    "is","are","was","were","be","been","being","have","has","had","do","does","did",
    "not","no","yes",
    "i","you","he","she","it","we","they","me","him","her","them","my","your","our","their","this","that","these","those",
    "just","really","very","kindly","please"
}

def _cf(s: str) -> str:
    return (s or "").casefold()

def _normalize(s: str) -> str:
    """lowercase, strip punctuation, collapse whitespace"""
    return " ".join(WORD_RE.findall(_cf(s)))


# ------------------------------
# Rulebook tokens (known phrases)
# ------------------------------

def load_rulebook_tokens(rulebook_path: Path) -> Set[str]:
    """
    Collect a broad set of 'known' tokens/phrases so we don't over-flag prompts.
    We include:
      - anchor blocks: synonyms, misspellings, synonyms_detailed[].token
      - top-level exact phrases (keys)
      - top-level combos[].all[] parts
      - keywords[anchor][] (if present)
    All values are normalized (no punctuation, lowercased).
    """
    with rulebook_path.open("r", encoding="utf-8") as f:
        rb = json.load(f)

    tokens: Set[str] = set()

    if isinstance(rb, dict):
        # Anchor blocks (dict keyed by anchor)
        for _k, block in rb.items():
            if not isinstance(block, dict):
                continue
            # arrays: synonyms / misspellings
            for key in ("synonyms", "misspellings"):
                for t in (block.get(key) or []):
                    if isinstance(t, str) and t.strip():
                        tokens.add(_normalize(t))
            # detailed synonyms: list of dicts with "token"
            for d in (block.get("synonyms_detailed") or []):
                if isinstance(d, dict):
                    t = d.get("token")
                    if isinstance(t, str) and t.strip():
                        tokens.add(_normalize(t))

        # exact map: { "phrase": "AnchorName", ... }
        exact = rb.get("exact")
        if isinstance(exact, dict):
            for phrase in exact.keys():
                if isinstance(phrase, str) and phrase.strip():
                    tokens.add(_normalize(phrase))

        # combos list: [{ "all": ["a","b"], ...}, ...]
        combos = rb.get("combos")
        if isinstance(combos, list):
            for combo in combos:
                if isinstance(combo, dict):
                    for part in (combo.get("all") or []):
                        if isinstance(part, str) and part.strip():
                            tokens.add(_normalize(part))

        # keywords bucket: { "AnchorName": ["k1","k2",...], ... }
        keywords = rb.get("keywords")
        if isinstance(keywords, dict):
            for arr in keywords.values():
                for t in (arr or []):
                    if isinstance(t, str) and t.strip():
                        tokens.add(_normalize(t))

    # prune empties
    tokens.discard("")
    return tokens


# ------------------------------
# Evidence iterator
# ------------------------------

def iter_evidence_prompts(paths: Iterable[str], fallback_only: bool) -> Iterable[str]:
    """
    Yield RAW prompts from evidence JSONL files.
    If fallback_only is True, only yield lines that have a non-empty 'fallback_reason'.
    """
    for pattern in paths:
        for fname in sorted(glob.glob(pattern)):
            p = Path(fname)
            if not p.exists():
                continue
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except Exception:
                        continue
                    prompt = (ev.get("prompt") or "").strip()
                    if len(prompt) < 4:  # ignore trivial noise
                        continue
                    if fallback_only and not (ev.get("fallback_reason") or "").strip():
                        continue
                    yield prompt


# ------------------------------
# Unknown mining
# ------------------------------

def _dedupe_adjacent(words: List[str]) -> List[str]:
    out: List[str] = []
    prev = None
    for w in words:
        if w and w != prev:
            out.append(w)
            prev = w
    return out

def find_unknowns(
    prompts: Iterable[str],
    known_tokens: Set[str],
    min_count: int = 3,
    stopwords: Set[str] | None = None,
) -> List[Dict[str, int]]:
    """
    Extract 'unknown' phrases from free-text prompts.

    Fix: emit single-word candidates after stopword removal so cases like
    'so so happy' -> 'happy' are not lost (test expectation).
    """
    sw = set(stopwords or DEFAULT_STOPWORDS)
    known = { _cf(k) for k in (known_tokens or set()) }

    counts: Counter[str] = Counter()

    for raw in prompts:
        s = _normalize(raw)
        if not s:
            continue
        words = [w for w in s.split() if w not in sw]
        words = _dedupe_adjacent(words)
        if not words:
            continue

        # --- conservative extraction ---
        # 1) unigrams (critical so 'happy' survives stopword stripping)
        for w in words:
            if w not in sw and w not in known and len(w) >= 2:
                counts[w] += 1

        # 2) (optional) light bigrams using the filtered words
        for i in range(len(words) - 1):
            bg = f"{words[i]} {words[i+1]}"
            if bg not in known:
                counts[bg] += 1

    # materialize ≥ min_count
    out = [{"phrase": k, "count": v} for k, v in counts.items() if v >= min_count]
    # stable ordering for diffs
    out.sort(key=lambda x: (-x["count"], x["phrase"]))
    return out


# ------------------------------
# CLI
# ------------------------------

def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Mine unknown phrases from evidence logs (noise-filtered)")
    ap.add_argument(
        "--logs",
        nargs="+",
        required=True,
        help="One or more glob patterns for evidence JSONL files (e.g., logs/evidence_*.jsonl)",
    )
    ap.add_argument(
        "--rules",
        required=True,
        help="Path to app/rules/emotion_keywords.json",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output file (candidates.json)",
    )
    ap.add_argument(
        "--min-count",
        type=int,
        default=3,
        help="Only include phrases seen at least this many times (default: 3)",
    )
    ap.add_argument(
        "--fallback-only",
        action="store_true",
        help="Only consider evidence lines that recorded a fallback_reason",
    )
    ap.add_argument(
        "--stopwords-file",
        help="Optional newline-delimited stopwords file (lowercased). If provided, extends the built-in list.",
    )
    args = ap.parse_args(argv)

    # Stopwords
    stopwords = set(DEFAULT_STOPWORDS)
    if args.stopwords_file:
        try:
            extra = Path(args.stopwords_file).read_text(encoding="utf-8").splitlines()
            stopwords.update([_normalize(x) for x in extra if x.strip()])
        except Exception:
            # Non-fatal; continue with defaults
            pass
    stopwords.discard("")  # safety

    rulebook = Path(args.rules)
    out_path = Path(args.out)

    known = load_rulebook_tokens(rulebook)
    prompts = list(iter_evidence_prompts(args.logs, args.fallback_only))
    unknowns = find_unknowns(prompts, known, args.min_count, stopwords)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(unknowns, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[mine_unknowns] scanned={len(prompts)} unique_unknowns={len(unknowns)} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
