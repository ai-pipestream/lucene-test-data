#!/usr/bin/env python3
"""Deduplicate embedding vec files in a sharded dataset.

Reads all shard vec files, removes exact-duplicate vectors (by raw byte hash),
re-shards the unique vectors evenly, and updates meta.json.

Usage:
    python3 dedup_vecs.py data/embeddings/wiki-1024-sentences
    python3 dedup_vecs.py data/embeddings/wiki-1024-sentences --dry-run
    python3 dedup_vecs.py data/embeddings/wiki-1024-sentences --no-backup
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import List


def shard_paths(dataset_dir: Path, num_shards: int) -> List[Path]:
    """Return ordered list of shard file paths."""
    if num_shards <= 1:
        p = dataset_dir / "docs.vec"
        if p.exists():
            return [p]
        # Fall back to shard-0
        return [dataset_dir / "docs-shard-0.vec"]
    return [dataset_dir / f"docs-shard-{i}.vec" for i in range(num_shards)]


def dedup(dataset_dir: Path, dry_run: bool, no_backup: bool) -> None:
    dataset_dir = Path(dataset_dir)
    meta_path = dataset_dir / "meta.json"
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    dim: int = meta["dim"]
    num_shards: int = meta.get("num_shards", 1)
    num_docs: int = meta["num_docs"]
    vec_bytes = dim * 4  # float32

    paths = shard_paths(dataset_dir, num_shards)
    for p in paths:
        if not p.exists():
            print(f"ERROR: shard file {p} not found", file=sys.stderr)
            sys.exit(1)

    # --- Pass 1: stream all shards, collect hashes of unique vectors ---
    seen = set()
    total = 0
    dupes = 0
    t0 = time.time()

    # In dry-run we only need to count. In real mode we also need to write.
    # For memory efficiency we do two passes in real mode:
    #   Pass 1: identify which vectors are unique (store hash set)
    #   Pass 2: stream again, write only first-seen vectors
    # But since we only keep hashes (not positions), a single-pass approach
    # that writes to a temp file is simpler and just as fast.

    tmp_path = dataset_dir / "_dedup_tmp.vec"

    if dry_run:
        for p in paths:
            with open(p, "rb") as f:
                while True:
                    raw = f.read(vec_bytes)
                    if len(raw) < vec_bytes:
                        break
                    total += 1
                    h = hashlib.md5(raw).digest()
                    if h in seen:
                        dupes += 1
                    else:
                        seen.add(h)
        elapsed = time.time() - t0
        unique = total - dupes
        print(f"[dry-run] {dataset_dir.name}: {total:,} vectors, {dupes:,} duplicates, {unique:,} unique  ({elapsed:.1f}s)")
        return

    # --- Real mode: single pass, stream to temp file ---
    unique_count = 0
    with open(tmp_path, "wb") as out:
        for p in paths:
            with open(p, "rb") as f:
                while True:
                    raw = f.read(vec_bytes)
                    if len(raw) < vec_bytes:
                        break
                    total += 1
                    h = hashlib.md5(raw).digest()
                    if h not in seen:
                        seen.add(h)
                        out.write(raw)
                        unique_count += 1
                    else:
                        dupes += 1

    elapsed = time.time() - t0
    print(f"{dataset_dir.name}: {total:,} vectors -> {unique_count:,} unique ({dupes:,} duplicates removed)  ({elapsed:.1f}s)")

    if dupes == 0:
        print("  No duplicates found, nothing to do.")
        tmp_path.unlink()
        return

    # --- Backup originals ---
    if not no_backup:
        print("  Backing up originals to *.pre-dedup ...")
        for p in paths:
            backup = p.with_suffix(p.suffix + ".pre-dedup")
            if not backup.exists():
                shutil.copy2(p, backup)
        meta_backup = meta_path.with_suffix(".json.pre-dedup")
        if not meta_backup.exists():
            shutil.copy2(meta_path, meta_backup)

    # --- Re-shard from temp file ---
    print(f"  Re-sharding {unique_count:,} vectors into {num_shards} shards ...")
    per_shard = unique_count // num_shards
    remainder = unique_count % num_shards

    new_shard_sizes: List[int] = []
    new_shard_offsets: List[int] = []
    offset = 0

    with open(tmp_path, "rb") as src:
        for s in range(num_shards):
            count = per_shard + (1 if s < remainder else 0)
            new_shard_sizes.append(count)
            new_shard_offsets.append(offset)
            offset += count

            shard_file = dataset_dir / (f"docs-shard-{s}.vec" if num_shards > 1 else "docs.vec")
            bytes_to_copy = count * vec_bytes
            with open(shard_file, "wb") as dst:
                remaining = bytes_to_copy
                while remaining > 0:
                    chunk_size = min(remaining, 8 * 1024 * 1024)
                    chunk = src.read(chunk_size)
                    if not chunk:
                        break
                    dst.write(chunk)
                    remaining -= len(chunk)

    tmp_path.unlink()

    # --- Update meta.json ---
    meta["num_docs"] = unique_count
    meta["shard_sizes"] = new_shard_sizes
    meta["shard_doc_offsets"] = new_shard_offsets

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    print(f"  Done. meta.json updated: num_docs={unique_count:,}")


def main():
    parser = argparse.ArgumentParser(description="Deduplicate embedding vec files in a sharded dataset.")
    parser.add_argument("dataset_dir", type=Path, help="Path to dataset directory containing meta.json and vec shards")
    parser.add_argument("--dry-run", action="store_true", help="Count duplicates without modifying anything")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating .pre-dedup backup files")
    args = parser.parse_args()

    dedup(args.dataset_dir, args.dry_run, args.no_backup)


if __name__ == "__main__":
    main()
