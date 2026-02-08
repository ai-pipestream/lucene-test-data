"""Write .vec files (luceneutil format: raw little-endian floats, no header) and manifest JSON."""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Sequence


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


def write_manifest(path: Path, data: Dict[str, Any]) -> None:
    """Write manifest JSON for index builder and data-dump metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
