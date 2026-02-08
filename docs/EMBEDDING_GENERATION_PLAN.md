# Embedding generation plan (embeddings only)

Goal: produce embedding files (`.vec` + metadata) for later indexing and recall tests. No index build in this phase.

## Reference: existing fast GPU embedding script

Outside this repo there is a script that already does Wikipedia → sentences → **DJL on GPU** with batched requests and is very fast. We use it as the pattern for our pipeline:

- **Source:** Wikipedia from a JSONL file (e.g. `simplewiki.json`: one JSON per line with `text`).
- **Chunking:** Sentence granularity (paragraph split by `\n\n`, then regex sentence-split).
- **Embedding:** POST to DJL Serving (e.g. `http://localhost:8091/predictions/<model>`), batch size ~100; GPU does the work so embedding is fast.
- **Output:** Text parts + binary embedding parts (e.g. 4 parts for 4 shards).

**All implementation for this plan lives in lucene-test-data.** We will build the embedding generator here to the same pattern (DJL on GPU, batched requests) and add:

- **Config:** YAML/CLI for source (wikipedia vs text_dir), URL, model, dim, batch_size, output paths.
- **Sources:** Wikipedia (e.g. JSONL path) and **text_dir** (directory of text files).
- **Granularity:** **sentence** and **paragraph**.
- **Output:** luceneutil `.vec` (little-endian, no header) + **manifest** (source, granularity, model, dim) for indexing and data-dump metadata.

---

## Pipeline (three phases)

1. **Phase 1 – Generate embeddings** (this plan): Python script + config → `.vec` files + manifest.
2. **Phase 2 – Build indices**: separate process reads `.vec` + manifest, builds Lucene indices (single/multi-shard).
3. **Phase 3 – Run tests**: run recall / leaf-pruning over (shard count × K), with and without collaborative HNSW.

---

## Source: generic (Wikipedia vs text directory)

Two mutually exclusive input modes:

| Source        | Config key        | Description |
|---------------|-------------------|-------------|
| **Wikipedia** | `source: wikipedia` | Use a “Wikipedia kit”: e.g. Hugging Face `wikipedia` dataset (e.g. `20231101.en`), or a pre-downloaded dump (e.g. one JSON/parquet file or a directory of text files exported from Wikipedia). |
| **Text files**| `source: text_dir`   | A directory of plain-text files (e.g. `.txt`). Each file is read; chunking (sentence vs paragraph) is applied to its content. |

Config should make the choice explicit and supply the path or dataset name so the **manifest can record it** (for the index field that tracks the source).

---

## Config parameters (Python script)

Suggested params (YAML or JSON; script can override with CLI or env).

### Source

- **`source`** (required): `wikipedia` | `text_dir`
- **`wikipedia`** (if source = wikipedia):
  - `dataset`: e.g. `wikipedia` + `config: "20231101.en"` (Hugging Face), or
  - `path`: path to a local dump (single file or directory of text/JSON/parquet)
  - Optional: `split` (e.g. `train`), `max_docs` (cap for testing)
- **`text_dir`** (if source = text_dir):
  - `path`: directory containing text files
  - Optional: `glob`: e.g. `"**/*.txt"`, `encoding`, `max_docs`

### Chunking (sentence vs paragraph)

- **`granularity`** (required for this run): `sentence` | `paragraph`
  - **sentence**: split by sentences (e.g. regex or `nltk`/`spacy`); one embedding per sentence (or per line if `line_per_unit: true`).
  - **paragraph**: split by paragraphs (e.g. `\n\n`, or fixed token count); one embedding per paragraph.
- Optional: `paragraph_delimiter`, `max_tokens_per_chunk`, `line_per_unit` (treat each line as one “sentence” when true).

### Embedding service (DJL)

- **`djl_url`**: base URL for DJL Serving (e.g. `http://localhost:8091`; host 8091 → container 8080).
- **`model_name`**: model ID as loaded in DJL (e.g. default model name in `/predictions/{model_name}`).
- **`dim`**: embedding dimension (must match model; used to write and validate `.vec`).
- Optional: `batch_size`, `timeout`, `max_length` (for BGE-M3-style APIs).

### Output

- **`output_dir`**: directory to write `.vec` and manifest (e.g. `data/embeddings`).
- **`output_stem`**: optional stem for filenames (default from source + granularity + model + dim), e.g.  
  `{source_id}-{granularity}-{model_id}-{dim}d` → `wikipedia-sentence-bge-m3-1024d-docs.vec`.
- **`num_query_vectors`**: number of vectors to hold out as queries (taken from doc set, e.g. first N or random); written to `*-queries.vec`.

### Manifest (for index and data dump)

Each run writes a **manifest** (e.g. `*-meta.json` or `manifest.json` next to the `.vec` files) so the index builder can store “source” and “granularity” (and other fields) in the index. Suggested fields:

- `source`: `wikipedia` | `text_dir`
- `source_path` or `source_dataset`: path or dataset id (e.g. `20231101.en`, `/path/to/text_dir`)
- `granularity`: `sentence` | `paragraph`
- `model_name`, `dim`
- `num_docs`, `num_query_vectors`
- `output_docs_vec`, `output_queries_vec` (filenames or paths)
- Optional: `created_at`, `config_hash`

---

## Four embedding sets (2×2)

To support “lower-dim / higher-dim” and “sentence / paragraph”:

| Set | Dimension | Granularity | Typical use |
|-----|-----------|-------------|-------------|
| 1   | Low (e.g. 384/768) | sentence   | Low-dim sentence embeddings |
| 2   | Low                | paragraph  | Low-dim paragraph embeddings |
| 3   | High (e.g. 1024)   | sentence   | High-dim sentence embeddings |
| 4   | High               | paragraph  | High-dim paragraph embeddings |

The script can:

- Accept **one** (model, granularity) per run and be invoked four times with different configs, or
- Accept a **list** of `(model_name, dim, granularity)` and run all four in one go.

Same source (Wikipedia or text_dir) should be used for all four so tests are comparable.

---

## Output file format

- **`.vec`**: raw little-endian floats, no header (same as luceneutil). Size = `num_vectors * dim * 4` bytes.
- **Docs**: one `.vec` file for all document vectors.
- **Queries**: one `.vec` file for the holdout query vectors (same dim, same format).
- **Manifest**: JSON with the fields above so that:
  - The index builder knows dimension and paths.
  - The index can store `source` and `granularity` (and optionally model/dim) as fields in the data dump.

---

## Later phases (out of scope here)

- **Index build**: reads `.vec` + manifest; builds one or more Lucene indices (single or multi-shard); stores manifest fields (e.g. `source`, `granularity`) in the index.
- **Test run**: for each embedding set, run (shard count × K), report recall and node visitation (and reduction vs non-collaborative); compare collaborative vs non-collaborative.

---

## Suggested script layout

```
lucene-test-data/
  scripts/
    generate_embeddings/
      README.md           # usage, config options, DJL setup
      config.example.yaml # all options documented
      generate_embeddings.py  # entrypoint: load config → load text → chunk → call DJL → write .vec + manifest
      sources/
        wikipedia.py     # load from HF dataset or local wikipedia dump
        text_dir.py      # load from directory of text files
      chunking.py        # sentence vs paragraph splitting
      djl_client.py      # batch embed via DJL Serving REST
      vec_io.py          # write .vec (little-endian) + read back for sanity check
  data/
    embeddings/         # output_dir for .vec + manifest per run
```

Config can be a single YAML file path (e.g. `config.example.yaml`) with optional overrides via CLI (`--source text_dir`, `--text_dir.path /path/to/files`) so we can support both Wikipedia kit and a directory of text files with the same script.
