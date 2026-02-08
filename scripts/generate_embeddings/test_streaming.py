#!/usr/bin/env python3
"""
Test the streaming vec writer with mock embeddings.
Verifies that streaming-to-disk + split-into-shards produces identical
output to the in-memory approach, using zero GPU and minimal RAM.

Usage:
    python3 scripts/generate_embeddings/test_streaming.py
    python3 scripts/generate_embeddings/test_streaming.py --num-vectors 100000 --dim 1024 --num-shards 16
"""
from __future__ import annotations

import argparse
import random
import shutil
import struct
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from vec_io import StreamingVecWriter, read_vec_file, write_vec_shards, write_manifest


def mock_batch(batch_size: int, dim: int, start_id: int) -> list[list[float]]:
    """Generate deterministic mock vectors: vec[i] = [i*0.001, i*0.001, ...] truncated to dim."""
    vectors = []
    for i in range(start_id, start_id + batch_size):
        val = (i % 10000) * 0.0001  # keep values small for float32 precision
        vectors.append([val] * dim)
    return vectors


def test_streaming(num_vectors: int, dim: int, num_shards: int, batch_size: int) -> bool:
    """Run streaming write, then verify shard contents match expectations."""
    tmp_dir = Path("/tmp/test_streaming_vec")
    streaming_dir = tmp_dir / "streaming"
    inmemory_dir = tmp_dir / "inmemory"

    # Clean up
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print(f"Testing: {num_vectors} vectors, dim={dim}, {num_shards} shards, batch={batch_size}")
    print()

    # ── Streaming approach ─────────────────────────────────────────
    print("1) Streaming write...")
    t0 = time.perf_counter()
    writer = StreamingVecWriter(streaming_dir, dim, num_shards)
    written = 0
    while written < num_vectors:
        bs = min(batch_size, num_vectors - written)
        batch = mock_batch(bs, dim, written)
        total = writer.append_batch(batch)
        written += bs
        if written % (batch_size * 10) == 0 or written == num_vectors:
            print(f"   Streamed {written}/{num_vectors}")
    shard_paths, shard_sizes, shard_offsets = writer.finalize()
    t_stream = time.perf_counter() - t0
    print(f"   Done in {t_stream:.2f}s")
    print(f"   Shards: {[p.name for p in shard_paths]}")
    print(f"   Sizes:  {shard_sizes}")
    print(f"   Offsets: {shard_offsets}")
    print()

    # ── In-memory approach (for comparison) ─────────────────────────
    print("2) In-memory write (comparison)...")
    t0 = time.perf_counter()
    all_vectors = mock_batch(num_vectors, dim, 0)
    im_paths, im_sizes, im_offsets = write_vec_shards(inmemory_dir, all_vectors, dim, num_shards)
    t_inmem = time.perf_counter() - t0
    del all_vectors  # free memory
    print(f"   Done in {t_inmem:.2f}s")
    print()

    # ── Verify ──────────────────────────────────────────────────────
    print("3) Verifying...")
    ok = True

    # Check sizes match
    if shard_sizes != im_sizes:
        print(f"   FAIL: shard_sizes differ: {shard_sizes} vs {im_sizes}")
        ok = False
    if shard_offsets != im_offsets:
        print(f"   FAIL: shard_offsets differ: {shard_offsets} vs {im_offsets}")
        ok = False
    if sum(shard_sizes) != num_vectors:
        print(f"   FAIL: shard sizes sum to {sum(shard_sizes)}, expected {num_vectors}")
        ok = False

    # Check file contents byte-for-byte
    for i in range(num_shards):
        sp = shard_paths[i]
        ip = im_paths[i]
        sb = sp.read_bytes()
        ib = ip.read_bytes()
        if sb != ib:
            print(f"   FAIL: shard {i} contents differ ({len(sb)} vs {len(ib)} bytes)")
            ok = False
        else:
            expected_bytes = shard_sizes[i] * dim * 4
            if len(sb) != expected_bytes:
                print(f"   FAIL: shard {i} size {len(sb)} != expected {expected_bytes}")
                ok = False

    # Spot-check first and last vector in first and last shard
    first_shard = read_vec_file(shard_paths[0], dim)
    if len(first_shard) != shard_sizes[0]:
        print(f"   FAIL: first shard has {len(first_shard)} vectors, expected {shard_sizes[0]}")
        ok = False
    else:
        expected_first = [0.0] * dim
        if first_shard[0] != expected_first:
            print(f"   FAIL: first vector mismatch")
            ok = False

    if ok:
        print(f"   ALL CHECKS PASSED")
    print()

    # ── Memory comparison ──────────────────────────────────────────
    stream_bytes = sum(p.stat().st_size for p in shard_paths)
    inmem_bytes = sum(p.stat().st_size for p in im_paths)
    print(f"Streaming: {t_stream:.2f}s, {stream_bytes / 1024 / 1024:.1f} MB on disk")
    print(f"In-memory: {t_inmem:.2f}s, {inmem_bytes / 1024 / 1024:.1f} MB on disk")
    print(f"Peak RAM (streaming): ~{batch_size * dim * 4 / 1024 / 1024:.1f} MB per batch")
    print(f"Peak RAM (in-memory): ~{num_vectors * dim * 36 / 1024 / 1024:.0f} MB (Python list overhead)")
    print()

    # Cleanup
    shutil.rmtree(tmp_dir)
    return ok


def main():
    ap = argparse.ArgumentParser(description="Test streaming vec writer with mock embeddings")
    ap.add_argument("--num-vectors", type=int, default=10000, help="Number of mock vectors")
    ap.add_argument("--dim", type=int, default=1024, help="Vector dimension")
    ap.add_argument("--num-shards", type=int, default=16, help="Number of shards")
    ap.add_argument("--batch-size", type=int, default=500, help="Vectors per mock batch")
    args = ap.parse_args()

    ok = test_streaming(args.num_vectors, args.dim, args.num_shards, args.batch_size)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
