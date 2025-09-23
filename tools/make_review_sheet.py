#!/usr/bin/env python3
"""
make_review_sheet.py — build a reviewer CSV from one or more proposals.json files.

Inputs
------
Each input JSON can be either:
  A) A bare list like:
     [{"phrase": "i am gutted", "count": 7}, ...]
  B) A richer list like:
     [{"phrase": "ecstatic for you", "count": 5,
       "suggested_anchor": "Encouragement/Positivity",
       "rationale": "congrats-like; positive polarity"}]
  C) Wrapped in a top-level object under "proposals":
     {"proposals": [ ... as A or B ... ]}

Output
------
CSV with headers compatible with tools/apply_tokens.py:
  phrase,decision,anchor,notes,list_type,intensity,polarity

- decision: left blank for the reviewer (expected values: approve|reject)
- anchor: prefilled if proposal has suggested_anchor/anchor
- notes: includes count and any rationale if present
- list_type: default "synonyms" (configurable via --default-list-type)
- intensity, polarity: empty (reviewer can set low|med|high and pos|neu|neg)

Typical
-------
  python tools/make_review_sheet.py \
    --in feeder/runs/2025-09-22/candidates.json \
    --in feeder/runs/2025-09-22/llm_proposals.json \
    --out feeder/runs/2025-09-22/review.csv \
    --min-count 3 \
    --limit 250

Safety
------
- Dedupe phrases case-insensitively.
- Keeps the *highest* observed count for duplicates.
- Merges rationales across files.
"""

from __future__ import annotations
import argparse, json, csv, sys
from pathlib import Path
from typing import Iterable, Tuple, Dict

DEFAULT_LIST_TYPE = "synonyms"
VALID_LIST_TYPES = {"synonyms", "misspellings"}

def _cf(s: str) -> str:
    return (s or "").casefold()

def load_proposals(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] skip {path}: {e}", file=sys.stderr)
        return []
    if isinstance(data, dict) and "proposals" in data:
        data = data["proposals"]
    if not isinstance(data, list):
        print(f"[warn] {path} is not a list/jsonl of proposals", file=sys.stderr)
        return []
    out = []
    for it in data:
        if not isinstance(it, dict):
            continue
        phrase = (it.get("phrase") or "").strip()
        if not phrase:
            continue
        # normalize aliases
        suggested_anchor = it.get("suggested_anchor") or it.get("anchor") or ""
        rationale = (it.get("rationale") or it.get("notes") or "").strip()
        count = it.get("count")
        try:
            count = int(count) if count is not None else 0
        except Exception:
            count = 0
        out.append({
            "phrase": phrase,
            "count": max(0, count),
            "suggested_anchor": suggested_anchor.strip(),
            "rationale": rationale,
        })
    return out

def merge_proposals(files: Iterable[Path]) -> list[dict]:
    by_phrase: Dict[str, dict] = {}
    for p in files:
        for it in load_proposals(p):
            k = _cf(it["phrase"])
            if k not in by_phrase:
                by_phrase[k] = it
            else:
                # keep max count; merge rationale (unique fragments)
                if it["count"] > by_phrase[k]["count"]:
                    by_phrase[k]["count"] = it["count"]
                if it["suggested_anchor"] and not by_phrase[k]["suggested_anchor"]:
                    by_phrase[k]["suggested_anchor"] = it["suggested_anchor"]
                if it["rationale"]:
                    rset = {r.strip() for r in (by_phrase[k]["rationale"] + " | " + it["rationale"]).split("|") if r.strip()}
                    by_phrase[k]["rationale"] = " | ".join(sorted(rset))
    return sorted(by_phrase.values(), key=lambda d: d["count"], reverse=True)

def main():
    ap = argparse.ArgumentParser(description="Create a reviewer CSV from proposals.json files.")
    ap.add_argument("--in", dest="inputs", nargs="+", required=True, help="One or more proposals.json paths")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--min-count", type=int, default=0, help="Keep rows with count >= min-count")
    ap.add_argument("--limit", type=int, default=0, help="Keep only the first N rows after filtering")
    ap.add_argument("--default-list-type", default=DEFAULT_LIST_TYPE, help="synonyms|misspellings (default: synonyms)")
    args = ap.parse_args()

    list_type = (args.default_list_type or DEFAULT_LIST_TYPE).strip().lower()
    if list_type not in VALID_LIST_TYPES:
        print(f"[warn] invalid --default-list-type {list_type!r}; using 'synonyms'", file=sys.stderr)
        list_type = "synonyms"

    files = [Path(p) for p in args.inputs]
    proposals = merge_proposals(files)
    if args.min_count > 0:
        proposals = [p for p in proposals if p["count"] >= args.min_count]
    if args.limit and args.limit > 0:
        proposals = proposals[:args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    headers = ["phrase", "decision", "anchor", "notes", "list_type", "intensity", "polarity"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for it in proposals:
            notes = []
            if it["count"]:
                notes.append(f"count={it['count']}")
            if it["rationale"]:
                notes.append(it["rationale"])
            w.writerow({
                "phrase": it["phrase"],
                "decision": "",  # reviewer fills: approve|reject
                "anchor": it["suggested_anchor"],
                "notes": " | ".join(notes),
                "list_type": list_type,
                "intensity": "",  # optional: low|med|high
                "polarity": "",   # optional: pos|neu|neg
            })

    print(f"[ok] wrote review sheet: {out_path} ({len(proposals)} rows)")

if __name__ == "__main__":
    main()
