import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root


def _write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["phrase", "decision", "anchor", "notes", "list_type", "intensity", "polarity"])
        w.writerows(rows)


def test_bad_list_type_hard_fails(tmp_path):
    # minimal rulebook (empty is fine)
    rb = tmp_path / "rb.json"
    rb.write_text("{}", encoding="utf-8")

    csv_path = tmp_path / "bad_list_type.csv"
    _write_csv(csv_path, [
        ["oops token", "approve", "Encouragement/Positivity", "x", "typo", "", ""],
    ])

    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "apply_tokens.py"),
         "--review", str(csv_path),
         "--rules", str(rb),
         "--dry-run"],
        capture_output=True, text=True
    )
    assert proc.returncode == 2
    assert "unknown list_type" in (proc.stdout + proc.stderr).lower()


def test_duplicate_phrase_warns_and_succeeds(tmp_path, capsys):
    rb = tmp_path / "rb.json"
    rb.write_text("{}", encoding="utf-8")

    csv_path = tmp_path / "dupe.csv"
    rows = [
        ["ecstatic for you", "approve", "Encouragement/Positivity", "x", "synonyms", "", ""],
        ["ecstatic for you", "approve", "Encouragement/Positivity", "x", "synonyms", "", ""],  # duplicate
    ]
    _write_csv(csv_path, rows)

    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "apply_tokens.py"),
         "--review", str(csv_path),
         "--rules", str(rb),
         "--dry-run"],
        capture_output=True, text=True
    )
    assert proc.returncode == 0
    assert "duplicate approved phrase" in proc.stdout.lower()
