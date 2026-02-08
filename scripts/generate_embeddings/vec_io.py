"""Write .vec files (luceneutil format: raw little-endian floats, no header) and manifest JSON."""
from __future__ import annotations

import json
import struct
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def write_vec_file(path: Path, vectors: Sequence[Sequence[float]], dim: int) -> None:
    """Write vectors to a .vec file. Each vector must have length dim."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = f"<{dim}f"
    with open(path, "wb") as f:
        for vec in vectors:
            if len(vec) != dim:
                raise ValueError(f"Vector length {len(vec)} != dim {dim}")
            f.write(struct.pack(fmt, *vec))
    return None


def read_vec_file(path: Path, dim: int) -> List[List[float]]:
    """Read vectors from a .vec file (sanity check / tests)."""
    path = Path(path)
    vec_size_bytes = dim * 4
    out = []
    with open(path, "rb") as f:
        while True:
            b = f.read(vec_size_bytes)
            if len(b) < vec_size_bytes:
                break
            out.append(list(struct.unpack(f"<{dim}f", b)))
    return out


def _write_shard(args: Tuple[Path, List[List[float]], int]) -> int:
    """Worker function for ProcessPoolExecutor: write one shard file."""
    path, vectors, dim = args
    write_vec_file(path, vectors, dim)
    return len(vectors)


def write_vec_shards(
    output_dir: Path,
    vectors: Sequence[Sequence[float]],
    dim: int,
    num_shards: int,
) -> Tuple[List[Path], List[int], List[int]]:
    """Split vectors into num_shards contiguous slices, write each as docs-shard-{i}.vec.

    Returns (shard_paths, shard_sizes, shard_doc_offsets).
    """
    n = len(vectors)
    per_shard = n // num_shards
    remainder = n % num_shards

    slices: List[Tuple[int, int]] = []
    start = 0
    for s in range(num_shards):
        count = per_shard + (1 if s < remainder else 0)
        slices.append((start, start + count))
        start += count

    shard_paths = [output_dir / f"docs-shard-{i}.vec" for i in range(num_shards)]
    shard_sizes = [end - begin for begin, end in slices]
    shard_offsets = [begin for begin, _ in slices]

    # Each worker needs a plain list (picklable); slice the vectors
    work = [
        (shard_paths[i], list(vectors[slices[i][0]:slices[i][1]]), dim)
        for i in range(num_shards)
    ]
    with ProcessPoolExecutor() as pool:
        list(pool.map(_write_shard, work))

    return shard_paths, shard_sizes, shard_offsets


def write_manifest(path: Path, data: Dict[str, Any]) -> None:
    """Write manifest JSON for index builder and data-dump metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
