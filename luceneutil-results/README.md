# Luceneutil KNN recall results (main vs PR)

Baseline comparison: **Lucene main** vs **Collaborative HNSW PR** using [luceneutil](https://github.com/mikemccand/luceneutil) KNN perf test.

- **Data:** `sentences_1024.bin` → 72,969 docs × 1024 dim, 10k query vectors, K=100, HNSW (cosine, maxConn=64, beamWidth=250).
- **Columns:** recall, latency(ms), netCPU, avgCpuCount, nDoc, topK, fanout, maxConn, beamWidth, quantized, visited, index(s), index_docs/s, force_merge(s), num_segments, index_size(MB), indexType.

Same index and exact-NN cache used for both runs (search-only comparison).

## Results in this folder

| Build | Summary file | Recall |
|-------|--------------|--------|
| **main** (apache/lucene main) | [main-summary.txt](main-summary.txt) | 0.996 |
| **PR** (feature/collaborative-hnsw-search) | [pr-summary.txt](pr-summary.txt) | 0.996 |

## Tool-logs (full run output)

Full logs live under `$BASE_DIR/tool-logs/` (e.g. `/work/opensearch-grpc-knn/tool-logs/`):

- **Main:** `knnPerfTest.py_20260208_105846*` (run with `external.lucene.repo` → lucene-main)
- **PR:** `knnPerfTest.py_20260208_085801*`, `knnPerfTest.py_20260208_105808*` (run with `external.lucene.repo` → lucene PR repo)
