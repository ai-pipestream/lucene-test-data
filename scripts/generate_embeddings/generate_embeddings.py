#!/usr/bin/env python3
"""
Generate embedding .vec files (luceneutil format) + manifest from Wikipedia or text_dir.
Uses DJL Serving (GPU) with configurable batch size.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

# Allow running from repo root or from this directory
_SCRIPT_DIR = Path(__file__).resolve().parent
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from chunking import chunk_text
from djl_client import embed_in_batches
from vec_io import write_vec_file, write_vec_shards, write_manifest

# Config: prefer YAML
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def load_config(path: Path) -> dict:
    if not HAS_YAML:
        raise RuntimeError("PyYAML required for config. Install with: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_source_id(cfg: dict) -> str:
    if cfg.get("source") == "wikipedia":
        p = cfg.get("wikipedia", {}).get("path") or "wikipedia"
        return Path(p).stem if isinstance(p, str) else "wikipedia"
    if cfg.get("source") == "text_dir":
        p = cfg.get("text_dir", {}).get("path") or "text_dir"
        return Path(p).name if isinstance(p, str) else "text_dir"
    return "unknown"


def load_texts(cfg: dict) -> list[str]:
    """Load raw document texts from configured source."""
    source = cfg.get("source")
    if not source:
        raise ValueError("config must set source: wikipedia or text_dir")

    if source == "wikipedia":
        from sources.wikipedia import load_wikipedia_all
        wiki = cfg.get("wikipedia", {})
        path = wiki.get("path")
        if not path:
            raise ValueError("wikipedia.path is required when source=wikipedia")
        return load_wikipedia_all(
            path,
            encoding=wiki.get("path_encoding", "utf-8"),
            max_docs=wiki.get("max_docs"),
            text_key=wiki.get("text_key", "text"),
        )

    if source == "text_dir":
        from sources.text_dir import load_text_dir_all
        td = cfg.get("text_dir", {})
        path = td.get("path")
        if not path:
            raise ValueError("text_dir.path is required when source=text_dir")
        return load_text_dir_all(
            path,
            glob=td.get("glob", "**/*.txt"),
            encoding=td.get("encoding", "utf-8"),
            max_docs=td.get("max_docs"),
        )

    raise ValueError(f"Unknown source: {source}")


def chunk_all_docs(doc_texts: list[str], cfg: dict) -> list[str]:
    """Chunk each document by granularity; return flat list of chunks."""
    granularity = cfg.get("granularity", "sentence")
    if granularity not in ("sentence", "paragraph"):
        raise ValueError(f"granularity must be sentence or paragraph, got {granularity!r}")
    opts = {
        "paragraph_delimiter": cfg.get("paragraph_delimiter", "\n\n"),
        "min_sentence_len": cfg.get("min_sentence_len", 10),
        "min_paragraph_len": cfg.get("min_paragraph_len", 20),
    }
    chunks = []
    for text in doc_texts:
        chunks.extend(chunk_text(text, granularity, **opts))
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate .vec embeddings + manifest via DJL")
    ap.add_argument("--config", "-c", type=Path, default=_SCRIPT_DIR / "config.example.yaml", help="YAML config path")
    ap.add_argument("--source", choices=("wikipedia", "text_dir"), help="Override source")
    ap.add_argument("--wikipedia-path", type=Path, help="Override wikipedia.path")
    ap.add_argument("--text-dir-path", type=Path, help="Override text_dir.path")
    ap.add_argument("--granularity", choices=("sentence", "paragraph"), help="Override granularity")
    ap.add_argument("--output-dir", type=Path, help="Override output.output_dir")
    ap.add_argument("--output-name", type=str, help="Override output.name (named subdir: e.g. unit-data, wiki-1024-sentences)")
    ap.add_argument("--batch-size", type=int, help="Override djl.batch_size")
    ap.add_argument("--max-docs", type=int, help="Override max_docs (wikipedia or text_dir)")
    ap.add_argument("--model-name", type=str, help="Override djl.model_name (e.g. all-MiniLM-L6-v2 or bge-m3)")
    ap.add_argument("--dim", type=int, help="Override djl.dim (e.g. 384 for MiniLM, 1024 for bge-m3)")
    ap.add_argument("--num-shards", type=int, default=1, help="Split docs.vec into N shard files (default: 1, no sharding)")
    args = ap.parse_args()

    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1
    cfg = load_config(args.config)

    if args.source:
        cfg["source"] = args.source
    if args.wikipedia_path:
        cfg.setdefault("wikipedia", {})["path"] = str(args.wikipedia_path)
    if args.text_dir_path:
        cfg.setdefault("text_dir", {})["path"] = str(args.text_dir_path)
    if args.granularity:
        cfg["granularity"] = args.granularity
    if args.output_dir:
        cfg.setdefault("output", {})["output_dir"] = str(args.output_dir)
    if args.output_name is not None:
        cfg.setdefault("output", {})["name"] = args.output_name
    if args.batch_size is not None:
        cfg.setdefault("djl", {})["batch_size"] = args.batch_size
    if args.max_docs is not None:
        cfg.setdefault("wikipedia", {})["max_docs"] = args.max_docs
        cfg.setdefault("text_dir", {})["max_docs"] = args.max_docs
    if args.model_name:
        cfg.setdefault("djl", {})["model_name"] = args.model_name
    if args.dim is not None:
        cfg.setdefault("djl", {})["dim"] = args.dim

    source = cfg.get("source")
    if not source:
        print("Missing source (wikipedia or text_dir). Set in config or --source.", file=sys.stderr)
        return 1

    print("Loading texts...")
    doc_texts = load_texts(cfg)
    print(f"  Loaded {len(doc_texts)} documents")

    print("Chunking...")
    chunks = chunk_all_docs(doc_texts, cfg)
    print(f"  Got {len(chunks)} chunks (granularity={cfg.get('granularity', 'sentence')})")

    if not chunks:
        print("No chunks to embed.", file=sys.stderr)
        return 1

    djl = cfg.get("djl", {})
    url = djl.get("url", "http://localhost:8091")
    model_name = djl.get("model_name", "bge_m3")
    dim = int(djl.get("dim", 1024))
    batch_size = int(djl.get("batch_size", 1000))
    timeout_sec = int(djl.get("timeout_sec", 120))

    print(f"Embedding via DJL ({url}, model={model_name}, dim={dim}, batch_size={batch_size})...")
    embeddings = embed_in_batches(
        url, model_name, chunks, dim,
        batch_size=batch_size,
        timeout_sec=timeout_sec,
        progress_interval=5,
    )
    print(f"  Got {len(embeddings)} vectors")

    out_cfg = cfg.get("output", {})
    base_output_dir = Path(out_cfg.get("output_dir", "data/embeddings"))
    num_queries = int(out_cfg.get("num_query_vectors", 5000))
    output_name = out_cfg.get("name")  # e.g. unit-data, wiki-1024-sentences

    if output_name:
        # All files go in a named subdir with fixed filenames
        output_dir = base_output_dir / output_name.strip()
        output_dir.mkdir(parents=True, exist_ok=True)
        docs_vec_path = output_dir / "docs.vec"
        queries_vec_path = output_dir / "queries.vec"
        meta_path = output_dir / "meta.json"
    else:
        stem = out_cfg.get("output_stem")
        if not stem:
            source_id = get_source_id(cfg)
            gran = cfg.get("granularity", "sentence")
            stem = f"{source_id}-{gran}-{model_name}-{dim}d"
        stem = stem.replace(" ", "-")
        output_dir = base_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        docs_vec_path = output_dir / f"{stem}-docs.vec"
        queries_vec_path = output_dir / f"{stem}-queries.vec"
        meta_path = output_dir / f"{stem}-meta.json"

    # Hold out first num_queries as queries (same as luceneutil convention)
    nq = min(num_queries, len(embeddings))
    query_vectors = embeddings[:nq]
    doc_vectors = embeddings  # full set as docs

    num_shards = args.num_shards

    if num_shards > 1:
        print(f"Writing {num_shards} shard files...")
        shard_paths, shard_sizes, shard_offsets = write_vec_shards(
            output_dir, doc_vectors, dim, num_shards
        )
        for sp in shard_paths:
            print(f"  Wrote {sp}")
    else:
        write_vec_file(docs_vec_path, doc_vectors, dim)
        print(f"Wrote {docs_vec_path}")

    write_vec_file(queries_vec_path, query_vectors, dim)

    source_path = ""
    if source == "wikipedia":
        source_path = cfg.get("wikipedia", {}).get("path", "")
    elif source == "text_dir":
        source_path = cfg.get("text_dir", {}).get("path", "")

    manifest = {
        "source": source,
        "source_path": source_path,
        "granularity": cfg.get("granularity", "sentence"),
        "model_name": model_name,
        "dim": dim,
        "num_docs": len(doc_vectors),
        "num_query_vectors": nq,
        "output_docs_vec": str(docs_vec_path) if num_shards <= 1 else "docs-shard-{i}.vec",
        "output_queries_vec": str(queries_vec_path),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if num_shards > 1:
        manifest["num_shards"] = num_shards
        manifest["shard_sizes"] = shard_sizes
        manifest["shard_doc_offsets"] = shard_offsets
    if output_name:
        manifest["dataset_name"] = output_name.strip()
    write_manifest(meta_path, manifest)

    print(f"Wrote {queries_vec_path}")
    print(f"Wrote {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
