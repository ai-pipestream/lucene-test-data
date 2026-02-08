# Generate embeddings (Phase 1)

Produces `.vec` files (luceneutil-compatible) and a manifest for later indexing. Supports **Wikipedia** or a **directory of text files**, and **sentence** or **paragraph** granularity.

## Quick start (one command)

From the **lucene-test-data** repo root:

```bash
./run-embeddings.sh
```

This automatically:
1. Starts DJL Serving with BGE-M3 (1024d) on GPU (or CPU fallback)
2. Downloads Simple English Wikipedia if `data/simplewiki.json` is missing
3. Generates sentence-level embeddings to `data/embeddings/`

Pass extra args to customize:

```bash
./run-embeddings.sh --granularity paragraph
./run-embeddings.sh --max-docs 5000              # quick test
./run-embeddings.sh --model-name all-MiniLM-L6-v2 --dim 384  # use MiniLM (needs load-models.sh first)
```

## Prerequisites (manual setup)

- Python 3.9+
- `pip install -r requirements.txt` (pyyaml, requests)
- DJL Serving running with BGE-M3:
  ```bash
  start-embedding-docker/start-djl.sh    # start container (BGE-M3 auto-loads)
  start-embedding-docker/restart-djl.sh  # or restart and wait for READY
  ```
- For Wikipedia: `data/simplewiki.json` (JSONL). If missing, run:
  ```bash
  python scripts/generate_embeddings/fetch_simplewiki.py
  ```
  See [DATA_SIMPLEWIKI.md](../../docs/DATA_SIMPLEWIKI.md) for details.

## Config

Copy `config.example.yaml` and set:

- **source**: `wikipedia` or `text_dir`
- **wikipedia**: `path` (local JSONL dump); optional `max_docs`
- **text_dir**: `path`, optional `glob`, `encoding`, `max_docs`
- **granularity**: `sentence` or `paragraph`
- **djl**: `url`, `model_name` (default `bge_m3`), `dim` (default 1024), `batch_size`
- **output**: `output_dir`, `num_query_vectors`, optional `output_stem`

**Note:** DJL names the auto-loaded BGE-M3 model `bge_m3` (underscore) from the directory name.

## Usage

```bash
# From lucene-test-data repo root:
python scripts/generate_embeddings/generate_embeddings.py -c scripts/generate_embeddings/config.example.yaml

# Override from CLI
python generate_embeddings.py --config config.yaml --source text_dir --text-dir-path /data/mytexts
python generate_embeddings.py --config config.yaml --wikipedia-path /path/to/simplewiki.json --batch-size 200
```

## Output

Under `output.output_dir`:

- `{stem}-docs.vec` -- document vectors (little-endian floats, no header)
- `{stem}-queries.vec` -- query vectors (holdout from docs)
- `{stem}-meta.json` -- manifest (source, granularity, model, dim, counts)

See [EMBEDDING_GENERATION_PLAN.md](../../docs/EMBEDDING_GENERATION_PLAN.md) for the full plan and four-set layout.
