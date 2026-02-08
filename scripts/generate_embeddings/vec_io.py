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


class StreamingVecWriter:
    """Stream vectors to a single .vec file as batches arrive, then split into shards.

    Usage:
        writer = StreamingVecWriter(output_dir, dim, num_shards)
        writer.append_batch(batch1)  # write immediately to temp file
        writer.append_batch(batch2)
        shard_paths, shard_sizes, shard_offsets = writer.finalize()  # split into shards
    """

    def __init__(self, output_dir: Path, dim: int, num_shards: int = 16):
        self.output_dir = Path(output_dir)
        self.dim = dim
        self.num_shards = num_shards
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._tmp_path = self.output_dir / "_docs_streaming.vec.tmp"
        self._fmt = f"<{dim}f"
        self._vec_bytes = dim * 4
        self._count = 0
        self._f = open(self._tmp_path, "wb")

    def append_batch(self, vectors: Sequence[Sequence[float]]) -> int:
        """Write a batch of vectors to the temp file. Returns running total."""
        fmt = self._fmt
        for vec in vectors:
            self._f.write(struct.pack(fmt, *vec))
            self._count += 1
        self._f.flush()
        return self._count

    def finalize(self) -> Tuple[List[Path], List[int], List[int]]:
        """Close temp file and split into num_shards shard files.

        Returns (shard_paths, shard_sizes, shard_doc_offsets).
        """
        self._f.close()
        n = self._count
        num_shards = self.num_shards

        if num_shards <= 1:
            # Just rename the temp file
            target = self.output_dir / "docs.vec"
            self._tmp_path.rename(target)
            return [target], [n], [0]

        per_shard = n // num_shards
        remainder = n % num_shards

        shard_paths: List[Path] = []
        shard_sizes: List[int] = []
        shard_offsets: List[int] = []
        offset = 0

        with open(self._tmp_path, "rb") as src:
            for s in range(num_shards):
                count = per_shard + (1 if s < remainder else 0)
                shard_path = self.output_dir / f"docs-shard-{s}.vec"
                bytes_to_copy = count * self._vec_bytes
                with open(shard_path, "wb") as dst:
                    remaining = bytes_to_copy
                    while remaining > 0:
                        chunk = src.read(min(remaining, 8 * 1024 * 1024))  # 8MB chunks
                        dst.write(chunk)
                        remaining -= len(chunk)
                shard_paths.append(shard_path)
                shard_sizes.append(count)
                shard_offsets.append(offset)
                offset += count

        self._tmp_path.unlink()
        return shard_paths, shard_sizes, shard_offsets

    @property
    def count(self) -> int:
        return self._count


def write_manifest(path: Path, data: Dict[str, Any]) -> None:
    """Write manifest JSON for index builder and data-dump metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
