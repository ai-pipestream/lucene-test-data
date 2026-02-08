# Roadmap: test data → index builder → test suite → CLI

Order of work for a full local test pipeline (folder method first, then indexer, then tester, then CLI).

---

## 1. Test data (folder method) — **current**

- **Source:** `index-builder/src/main/resources/test-texts/` (Alice in Wonderland + Pride and Prejudice).
- **Configs:** Four YAML configs in `scripts/generate_embeddings/configs/` (sentence/paragraph × MiniLM 384d / bge-m3 1024d).
- **Generate:** From repo root:
  ```bash
  ./scripts/generate_embeddings/run_test_text_embeddings.sh
  ```
- **Output:** `index-builder/src/main/resources/embeddings/*.vec` and `*-meta.json` (small, committed). Good for local testing without large Wikipedia data.

---

## 2. Index builder (Phase 2)

- **Input:** `*-docs.vec`, `*-queries.vec`, `*-meta.json` (from Phase 1 or from classpath/resources).
- **Output:** Lucene index (single or N shards), e.g. `KnnFloatVectorField`, optional stored fields for source/granularity.
- **Scope:** Read .vec with NIO (little-endian floats), build index; support classpath or given path. Later: optional multi-shard split for shard tests.

---

## 3. Test suite (recall / shard × K)

- **Runs:** Multiple (K values × shard counts × Lucene jar variants).
- **Metrics:** luceneutil-style SUMMARY (recall, latency, nDoc, topK, etc.) + extra metadata (e.g. processing checks, config).
- **Ground truth:** Exact NN from .vec (or precomputed) for recall.
- **Output:** Tabular results + optional JSON/HTML for comparison across jars.

---

## 4. Fancy CLI

- Single entrypoint (e.g. `run_tests` or Gradle task) with subcommands or flags:
  - Generate embeddings (folder or wikipedia, config path).
  - Build index from manifest / .vec path.
  - Run test suite (K list, shard list, Lucene jar path(s)).
  - Output path, format (text, JSON).
- Can wrap the Python embedding script + Java index builder + Java (or script) test runner.

---

## References

- **Embedding plan:** `docs/EMBEDDING_GENERATION_PLAN.md`
- **Configs:** `scripts/generate_embeddings/configs/README.md`
- **Index builder:** `index-builder/README.md`
