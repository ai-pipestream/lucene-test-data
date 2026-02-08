"""Batch embed text via DJL Serving REST API (GPU)."""
from __future__ import annotations

import time
from typing import List

import requests


def embed_batch(
    url: str,
    model_name: str,
    texts: List[str],
    dim: int,
    timeout_sec: int = 120,
) -> List[List[float]]:
    """
    POST a list of texts to DJL /predictions/{model_name}. Expects JSON body as list of strings.
    Returns list of embedding vectors; pads or truncates to dim if needed.
    """
    if not texts:
        return []
    endpoint = f"{url.rstrip('/')}/predictions/{model_name}"
    try:
        resp = requests.post(endpoint, json=texts, timeout=timeout_sec)
        resp.raise_for_status()
        batch_embs = resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"DJL request failed: {e}") from e

    out = []
    for emb in batch_embs:
        if isinstance(emb, dict) and "embedding" in emb:
            vec = emb["embedding"]
        elif isinstance(emb, list):
            vec = emb
        else:
            raise ValueError(f"Unexpected embedding shape: {type(emb)}")
        # Pad or truncate to dim
        if len(vec) < dim:
            vec = list(vec) + [0.0] * (dim - len(vec))
        elif len(vec) > dim:
            vec = vec[:dim]
        out.append(vec)
    return out


def embed_in_batches(
    url: str,
    model_name: str,
    texts: List[str],
    dim: int,
    batch_size: int = 1000,
    timeout_sec: int = 120,
    progress_interval: int = 10,
) -> List[List[float]]:
    """Embed all texts in batches; print progress every progress_interval batches."""
    all_embeddings = []
    total = len(texts)
    start = time.perf_counter()
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        batch_embs = embed_batch(url, model_name, batch, dim, timeout_sec=timeout_sec)
        all_embeddings.extend(batch_embs)
        batch_num = i // batch_size + 1
        if progress_interval and batch_num % progress_interval == 0:
            elapsed = time.perf_counter() - start
            print(f"  Embedded {len(all_embeddings)}/{total} ({batch_num} batches, {elapsed:.1f}s)")
    return all_embeddings
