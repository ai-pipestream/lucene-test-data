#!/usr/bin/env python3
"""
Ensure data/simplewiki.json exists. If not, download Simple English Wikipedia
from Hugging Face (wikipedia, 20220301.simple) and write JSONL with one {"text": ...} per line.
Run from lucene-test-data repo root, or set OUTPUT_PATH.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Default: data/simplewiki.json relative to repo root (parent of scripts/generate_embeddings)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = _REPO_ROOT / "data" / "simplewiki.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch Simple Wikipedia JSONL if missing")
    ap.add_argument(
        "-o", "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if file exists",
    )
    args = ap.parse_args()

    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"Already exists: {out}")
        return 0

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install Hugging Face datasets: pip install datasets", file=sys.stderr)
        return 1

    print("Loading Simple English Wikipedia from Hugging Face (20220301.simple)...")
    ds = load_dataset("wikipedia", "20220301.simple", split="train", trust_remote_code=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing JSONL to {out}...")
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            text = row.get("text") or ""
            if text.strip():
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {out} ({out.stat().st_size / (1024*1024):.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
