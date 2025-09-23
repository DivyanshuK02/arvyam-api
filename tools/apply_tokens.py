#!/usr/bin/env python3
"""
apply_tokens.py — apply human-approved phrases to the rulebook (additive only).

CSV header (required):
  phrase,decision,anchor,notes[,list_type][,intensity][,polarity]

Enums (hard-fail on unknown):
  - anchor: one of the 8 canonical anchors (exact text)
  - decision: approve | skip     (# 'reject' is treated as 'skip' with a warning for back-compat)
  - list_type: synonyms | misspellings | synonyms_detailed
  - intensity: low | med | high | ""   (empty means unspecified)
  - polarity : pos | neu | neg | ""    (empty means unspecified)

Behavior:
  - Additive only (no deletions)
  - First approved row for a phrase wins within a single CSV; later duplicates are ignored with a warning
  - If intensity or polarity is provided (non-empty), row is stored in synonyms_detailed[]
    (explicit list_type='synonyms_detailed' is also supported)
  - Emits a timestamped .bak before writing
  - --dry-run prints the plan and does not write the file
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

# --- Canonical enums ---------------------------------------------------------

ANCHORS = {
    "Affection/Support",
    "Loyalty/Dependability",
    "Encouragement/Positivity",
    "Strength/Resilience",
    "Intellect/Wisdom",
    "Adventurous/Creativity",
    "Selflessness/Generosity",
    "Fun/Humor",
}

VALID_DECISIONS = {"approve", "skip"}
VALID_LIST_TYPES = {"synonyms", "misspellings", "synonyms_detailed"}
VALID_INTENSITY = {"low", "med", "high"}
VALID_POLARITY = {"pos", "neu", "neg"}


# --- Helpers -----------------------------------------------------------------

def _cf(s: str) -> str:
    return (s or "").casefold()


def die(msg: str, code: int = 2) -> None:
    print(f"[error] {msg}", file=sys.stderr)
    sys.exit(code)


def load_rulebook(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def backup(path: Path) -> Path:
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    bak = path.with_suffix(path.suffix + f".{ts}.bak")
    shutil.copy2(path, bak)
    return bak


def ensure_lists(block: dict) -> None:
    block.setdefault("synonyms", [])
    block.setdefault("misspellings", [])
    block.setdefault("synonyms_detailed", [])


def sort_ci(strings: List[str]) -> List[str]:
    return sorted(strings, key=lambda s: (s or "").lower())


def sort_detailed(entries: List[dict]) -> List[dict]:
    return sorted(entries, key=lambda d: (d.get("token") or "").lower())


# --- CLI ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Apply approved phrases to emotion_keywords.json (add-only)")
    ap.add_argument("--review", required=True, help="CSV with phrase decisions")
    ap.add_argument("--rules", default="app/rules/emotion_keywords.json", help="Path to rulebook JSON")
    ap.add_argument("--dry-run", action="store_true", help="Print changes only; do not write")
    args = ap.parse_args()

    rules_path = Path(args.rules)
    rb = load_rulebook(rules_path)

    # Build quick lookup for existing strings (casefold) per anchor + list
    def existing_set(block: dict, list_key: str) -> set[str]:
        return {_cf(s) for s in (block.get(list_key) or []) if isinstance(s, str)}

    # Staged changes per anchor
    staged: Dict[str, Dict[str, list]] = {
        a: {"synonyms": [], "misspellings": [], "synonyms_detailed": []} for a in ANCHORS
    }

    # Track duplicates within this CSV (first approved wins)
    seen_cf_approved: set[str] = set()

    added_rows = 0

    # Read and validate CSV
    with open(args.review, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        required = {"phrase", "decision", "anchor"}
        missing = required - set(rdr.fieldnames or [])
        if missing:
            die(f"review CSV missing columns: {sorted(missing)}")

        for i, row in enumerate(rdr, start=2):  # start=2 because header is line 1
            raw_phrase = (row.get("phrase") or "").strip()
            raw_anchor = (row.get("anchor") or "").strip()
            raw_decision = (row.get("decision") or "").strip().lower()
            raw_list_type = (row.get("list_type") or "synonyms").strip().lower()
            raw_intensity = (row.get("intensity") or "").strip().lower()
            raw_polarity = (row.get("polarity") or "").strip().lower()

            if not raw_phrase:
                # skip empty phrase rows silently
                continue

            # Decision handling (back-compat for 'reject')
            if raw_decision == "reject":
                print(f"[warn] line {i}: decision 'reject' is deprecated; treating as 'skip'")
                raw_decision = "skip"
            if raw_decision not in VALID_DECISIONS:
                die(f"line {i}: unknown decision {raw_decision!r} (expected one of {sorted(VALID_DECISIONS)})")

            if raw_decision == "skip":
                continue  # not applied

            # Anchor must be exact
            if raw_anchor not in ANCHORS:
                die(f"line {i}: unknown anchor {raw_anchor!r}")

            # Duplicate within this CSV
            pcf = _cf(raw_phrase)
            if pcf in seen_cf_approved:
                print(f"[warn] line {i}: duplicate approved phrase {raw_phrase!r} → ignoring (first wins)")
                continue
            seen_cf_approved.add(pcf)

            # list_type validation
            if raw_list_type not in VALID_LIST_TYPES:
                die(f"line {i}: unknown list_type {raw_list_type!r} (expected one of {sorted(VALID_LIST_TYPES)})")

            # intensity/polarity validation (allow empty)
            if raw_intensity and raw_intensity not in VALID_INTENSITY:
                die(f"line {i}: unknown intensity {raw_intensity!r} (expected one of {sorted(VALID_INTENSITY)} or empty)")
            if raw_polarity and raw_polarity not in VALID_POLARITY:
                die(f"line {i}: unknown polarity {raw_polarity!r} (expected one of {sorted(VALID_POLARITY)} or empty)")

            # Determine storage: detailed if explicit list_type OR any intensity/polarity provided
            use_detailed = (raw_list_type == "synonyms_detailed") or bool(raw_intensity or raw_polarity)

            if use_detailed:
                staged[raw_anchor]["synonyms_detailed"].append(
                    {"token": raw_phrase, "intensity": (raw_intensity or None), "polarity": (raw_polarity or None)}
                )
            else:
                staged[raw_anchor][raw_list_type].append(raw_phrase)

            added_rows += 1

    if not added_rows:
        print("[info] no approved rows to apply; nothing to do.")
        return

    # Apply staged changes to rulebook (additive + dedupe)
    for anchor, adds in staged.items():
        if not any(adds.values()):
            continue
        block = rb.setdefault(anchor, {})
        ensure_lists(block)

        # Simple lists: synonyms / misspellings
        for list_key in ("synonyms", "misspellings"):
            if not adds[list_key]:
                continue
            existing = existing_set(block, list_key)
            for s in adds[list_key]:
                if _cf(s) not in existing:
                    block[list_key].append(s)
                    existing.add(_cf(s))
            # Deduplicate and sort deterministically
            dedup = []
            seen = set()
            for s in block[list_key]:
                k = _cf(s)
                if k not in seen:
                    dedup.append(s)
                    seen.add(k)
            block[list_key] = sort_ci(dedup)

        # Detailed list: synonyms_detailed (dedupe by token casefold, then sort by token)
        if adds["synonyms_detailed"]:
            existing_tokens = {
                _cf(d.get("token", "")) for d in (block.get("synonyms_detailed") or []) if isinstance(d, dict)
            }
            for d in adds["synonyms_detailed"]:
                tok = _cf(d.get("token", ""))
                if tok and tok not in existing_tokens:
                    block["synonyms_detailed"].append(d)
                    existing_tokens.add(tok)
            # Deduplicate in place (by token) then sort
            dedup_detail = []
            seen_tok = set()
            for d in block["synonyms_detailed"]:
                tok = _cf(d.get("token", ""))
                if tok and tok not in seen_tok:
                    dedup_detail.append(d)
                    seen_tok.add(tok)
            block["synonyms_detailed"] = sort_detailed(dedup_detail)

    # Plan summary
    print("[plan] staged updates:")
    for a in ANCHORS:
        adds = staged[a]
        if any(adds.values()):
            print(
                f"  - {a}: +{len(adds['synonyms'])} synonyms,"
                f" +{len(adds['misspellings'])} misspellings,"
                f" +{len(adds['synonyms_detailed'])} detailed"
            )

    if args.dry_run:
        print("[dry-run] not writing changes.")
        return

    bak = backup(rules_path)
    with rules_path.open("w", encoding="utf-8") as f:
        json.dump(rb, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[ok] rulebook updated: {rules_path} (backup: {bak.name})")


if __name__ == "__main__":
    main()
