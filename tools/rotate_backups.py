#!/usr/bin/env python3
import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Delete old .bak files, keep the newest N.")
    p.add_argument("--dir", default="app/rules", help="Folder with backups")
    p.add_argument("--pattern", default="emotion_keywords.json.*.bak", help="Backup filename pattern")
    p.add_argument("--keep", type=int, default=10, help="How many newest backups to keep")
    args = p.parse_args()

    d = Path(args.dir)
    files = sorted(d.glob(args.pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    if not files:
        print("[rotate_backups] No backups found — nothing to do.")
        return

    to_delete = files[args.keep:]
    for f in to_delete:
        try:
            f.unlink()
            print(f"[rotate_backups] deleted: {f.name}")
        except Exception as e:
            print(f"[rotate_backups] could not delete {f.name}: {e}")

    print(f"[rotate_backups] kept {min(len(files), args.keep)} newest; removed {len(to_delete)} old.")

if __name__ == "__main__":
    main()
