#!/usr/bin/env python3
# tools/mine_unknowns.py
#
# Mine "unknown" user phrases from SELECTION_EVIDENCE logs for the offline feeder.
# This script is OFFLINE-ONLY and never runs in the request path.

import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Set


def _casefold(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def load_rulebook_tokens(rulebook_path: Path) -> Set[str]:
    """
    Collect a broad set of 'known' tokens/phrases from the rulebook so we don't
    over-flag prompts. We include:
      - anchor blocks: synonyms, misspellings, synonyms_detailed[].token
      - top-level exact phrases
      - top-level combos[].all[]
      - keywords[anchor][] (if present)
    """
    with rulebook_path.open("r", encoding="utf-8") as f:
        rb = json.load(f)

    tokens: Set[str] = set()

    # Anchor blocks (dict keyed by anchor) – be defensive about shape
    if isinstance(rb, dict):
        for _k, block in rb.items():
            if not isinstance(block, dict):
                continue
            # synonyms / misspellings arrays
            for key in ("synonyms", "misspellings"):
                for t in (block.get(key) or []):
                    if isinstance(t, str) and t.strip():
                        tokens.add(_casefold(t))
            # synonyms_detailed: list of dicts with a "token" field
            for d in (block.get("synonyms_detailed") or []):
                if isinstance(d, dict):
                    t = d.get("token")
                    if isinstance(t, str) and t.strip():
                        tokens.add(_casefold(t))

    # Top-level exact map: { "phrase": "AnchorName", ... }
    exact = rb.get("exact") if isinstance(rb, dict) else None
    if isinstance(exact, dict):
        for phrase in exact.keys():
            if isinstance(phrase, str) and phrase.strip():
                tokens.add(_casefold(phrase))

    # Top-level combos list: [{ "all": ["a","b"], ... }, ...]
    combos = rb.get("combos") if isinstance(rb, dict) else None
    if isinstance(combos, list):
        for combo in combos:
            if isinstance(combo, dict):
                for part in (combo.get("all") or []):
                    if isinstance(part, str) and part.strip():
                        tokens.add(_casefold(part))

    # keywords bucket: { "AnchorName": ["k1","k2",...], ... }
    keywords = rb.get("keywords") if isinstance(rb, dict) else None
    if isinstance(keywords, dict):
        for arr in keywords.values():
            for t in (arr or []):
                if isinstance(t, str) and t.strip():
                    tokens.add(_casefold(t))

    return tokens


def iter_evidence_prompts(paths: Iterable[str], fallback_only: bool) -> Iterable[str]:
    """
    Yield normalized prompts from evidence JSONL files.
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

                    if fallback_only:
                        fr = (ev.get("fallback_reason") or "").strip()
                        if not fr:
                            continue

                    yield _casefold(prompt)


def find_unknowns(
    prompts: Iterable[str],
    known_tokens: Set[str],
    min_count: int = 3,
) -> List[dict]:
    """
    Keep prompts that don't contain any known token/phrase (substring match),
    and aggregate counts. Return a list of {phrase, count} sorted by count desc.
    """
    ctr: Counter[str] = Counter()
    for pr in prompts:
        # If ANY known token appears as substring, we treat the prompt as "covered"
        if any(tok and tok in pr for tok in known_tokens):
            continue
        ctr[pr] += 1

    results = [
        {"phrase": phrase, "count": count}
        for phrase, count in ctr.items()
        if count >= max(1, int(min_count))
    ]
    results.sort(key=lambda x: (-x["count"], x["phrase"]))
    return results


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Mine unknown phrases from evidence logs")
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
    args = ap.parse_args(argv)

    rulebook = Path(args.rules)
    out_path = Path(args.out)

    known = load_rulebook_tokens(rulebook)
    prompts = list(iter_evidence_prompts(args.logs, args.fallback_only))
    unknowns = find_unknowns(prompts, known, args.min_count)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(unknowns, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[mine_unknowns] scanned={len(prompts)} unique_unknowns={len(unknowns)} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
