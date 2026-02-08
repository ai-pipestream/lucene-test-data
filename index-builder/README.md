# Index builder (Phase 2)

Builds Lucene indices from the `.vec` files and manifests produced by the embedding script (Phase 1).

## Dependencies (minimal)

| Library | Purpose |
|--------|---------|
| **Lucene core** | `org.apache.lucene:lucene-core` – `IndexWriter`, `Document`, `KnnFloatVectorField`, `VectorSimilarityFunction`, `FSDirectory`, HNSW indexing. |
| **Gson** | `com.google.code.gson:gson` – Read `*-meta.json` manifests (dim, source, granularity, paths). Single small jar; no transitive bloat. |
| **JUnit 5** (optional) | For tests. Omit if you don’t need them. |
| **SLF4J + simple** (optional) | For logging. Can use `System.out` instead. |

Nothing else required: .vec is raw little-endian floats (Java NIO), and Lucene core contains vector/HNSW support.

## Optional later

- **lucene-misc** – Only if you want `BPReorderingMergePolicy` / block pivot reordering like luceneutil; not needed for a first version.
- **Picocli / Commons CLI** – If you want a fancy CLI; otherwise `main(String[] args)` is enough.
- **Local Lucene build** – To test against your collaborative HNSW PR, point Gradle at a local Lucene repo (like luceneutil’s `external.lucene.repo`) instead of Maven Central.

## Inputs

- `*-docs.vec` – Document vectors (little-endian float32, no header).
- `*-meta.json` – Manifest with `dim`, `output_docs_vec`, `output_queries_vec`, `source`, `granularity`, etc.
- Optional: store `source` / `granularity` in the index (e.g. as stored fields) for the data dump.

## Output

- Lucene index directory (single index per run, or multiple for multi-shard later).

## Using a local Lucene build (e.g. collaborative HNSW PR)

To test against your PR branch instead of Maven Central, in `build.gradle.kts` replace the `lucene-core` dependency with a local path or `fileTree`, e.g.:

```kotlin
// Local Lucene (e.g. /path/to/lucene)
val luceneLibs = fileTree("path/to/lucene/core/build/libs").matching { include("*.jar") }
implementation(luceneLibs)
```

Or use a Gradle property (e.g. `external.lucene.repo`) like luceneutil.

## Build / run

```bash
./gradlew jar
./gradlew test
```

Run the index builder (dataset by name or path, output index dir):

```bash
# From index-builder dir, using datasets in build/resources/main (after processResources)
./gradlew run --args="--dataset unit-data-1024-sentence --output build/index-unit-1024 --base build/resources/main"

# With absolute path to dataset dir
./gradlew run --args="--dataset /path/to/embeddings/unit-data-1024-sentence --output /path/to/index"

# Build 4 shard indices (for shard recall tests)
./gradlew run --args="--dataset unit-data-1024-sentence --output build/index-4shards --base build/resources/main --num-shards 4"
# → creates build/index-4shards/shard-0, shard-1, shard-2, shard-3

# Large data under data/embeddings (e.g. wiki-1024-sentences): 8 shards, batch 1000
./gradlew run --args="--dataset wiki-1024-sentences --output build/index-wiki-8shards --base ../data --num-shards 8 --batch-size 1000"
# From repo root, --base data makes it look for data/embeddings/<dataset>/
```

**Options:**  
- `--num-shards N` – split into N indices (contiguous doc id ranges). Output: `<output>/shard-0`, … Each doc stores global `id` for merge/recall.  
- `--batch-size N` – documents per `addDocuments` batch (default 1000). Larger = faster indexing, more RAM.

Datasets: each is a directory with `docs.vec`, `queries.vec`, `meta.json`. Built-in names (from resources): `unit-data-1024-sentence`, `unit-data-1024-paragraph`. When MiniLM is available, add `unit-data-384-sentence`, `unit-data-384-paragraph` via the embedding script.
