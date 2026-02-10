#!/bin/bash
# Run the same 8-shard test as baseline-8shards-with-recall.txt but with PR Lucene JAR + --collaborative.
# Baseline is already saved; this run produces the PR comparison (recall should match; latency/visits may differ).
#
# Uses --query-threads and --search-threads so CPU is utilized. Without parallel queries,
# the single-threaded exact NN over 2.1M docs dominates and the run takes many hours.
#
# From index-builder:
#   ./run-pr-8shard-test.sh
# Or from repo root:
#   index-builder/run-pr-8shard-test.sh
# Override: QUERY_THREADS=64 SEARCH_THREADS=64 ./run-pr-8shard-test.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LUCENE_JAR="${LUCENE_JAR:-/work/opensearch-grpc-knn/lucene/lucene/core/build/libs/lucene-core-11.0.0-SNAPSHOT.jar}"
# Backward codecs so Lucene 11 can read indexes created with Lucene 10 (Lucene103)
LUCENE_BACKWARD_CODECS_JAR="${LUCENE_BACKWARD_CODECS_JAR:-/work/opensearch-grpc-knn/lucene/lucene/backward-codecs/build/libs/lucene-backward-codecs-11.0.0-SNAPSHOT.jar}"
# 32 threads (query + search); ~30x faster than single-threaded 8hr run
QUERY_THREADS="${QUERY_THREADS:-32}"
SEARCH_THREADS="${SEARCH_THREADS:-32}"
SHARDS="/work/opensearch-grpc-knn/shards_to_test/index-simplewiki-8shards"
QUERIES="/work/embedding-archive-data/embeddings/simplewiki-sentence-bge_m3-1024d-queries.vec"
DOCS="/work/embedding-archive-data/embeddings/simplewiki-sentence-bge_m3-1024d-docs.vec"
DIM=1024

if [ ! -f "$LUCENE_JAR" ]; then
    echo "PR Lucene JAR not found: $LUCENE_JAR"
    echo "Build it with: cd /work/opensearch-grpc-knn/lucene && ./gradlew :lucene:core:jar"
    echo "Or set LUCENE_JAR=/path/to/lucene-core-*.jar"
    exit 1
fi

echo "Using Lucene JAR: $LUCENE_JAR"
echo "Shards: $SHARDS  query-threads=$QUERY_THREADS  search-threads=$SEARCH_THREADS  --collaborative"
echo ""

./gradlew -q runShardTest \
  -PluceneJar="$LUCENE_JAR" \
  -PluceneBackwardCodecsJar="$LUCENE_BACKWARD_CODECS_JAR" \
  --args="--shards $SHARDS --queries $QUERIES --docs $DOCS --dim $DIM --collaborative --query-threads $QUERY_THREADS --search-threads $SEARCH_THREADS --progress" \
  "$@"
