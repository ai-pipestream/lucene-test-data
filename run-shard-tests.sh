#!/bin/bash
# Run KNN shard tests with recall (vs exact NN) for each dataset that has indices + embeddings.
# Uses index-builder's RunShardTest; no luceneutil needed. Recall = |retrieved ∩ exact| / K.
#
# Prerequisites:
#   - Indices built: ./run-all-indices.sh
#   - Embedding dirs have queries.vec and docs.vec or docs-shard-*.vec
#
# Usage:
#   ./run-shard-tests.sh              # test all datasets, 8 shards
#   ./run-shard-tests.sh --shards 4   # use shards-4 instead of shards-8
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

GRADLE="index-builder/gradlew -p index-builder"
EMBEDDINGS_BASE="$SCRIPT_DIR/data/embeddings"
INDICES_BASE="$SCRIPT_DIR/data/indices"
SHARDS=8

while [[ $# -gt 0 ]]; do
    case "$1" in
        --shards) SHARDS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--shards N]"
            echo "  Runs RunShardTest with --docs so recall is computed. Default: shards-8."
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

DATASETS=(
    "wiki-1024-sentences"
    "wiki-1024-paragraphs"
    "wiki-384-sentences"
    "wiki-384-paragraphs"
)

echo "═══════════════════════════════════════════════════════"
echo "  Shard KNN tests (recall + latency), shards=$SHARDS"
echo "═══════════════════════════════════════════════════════"

for ds in "${DATASETS[@]}"; do
    embed_dir="$EMBEDDINGS_BASE/$ds"
    index_dir="$INDICES_BASE/$ds/shards-$SHARDS"
    meta="$embed_dir/meta.json"
    if [ ! -f "$meta" ]; then
        echo ""
        echo "SKIP $ds (no $meta)"
        continue
    fi
    if [ ! -f "$embed_dir/queries.vec" ]; then
        echo ""
        echo "SKIP $ds (no queries.vec)"
        continue
    fi
    if [ ! -d "$index_dir/shard-0" ]; then
        echo ""
        echo "SKIP $ds (no $index_dir)"
        continue
    fi
    dim=$(python3 -c "import json; print(json.load(open('$meta'))['dim'])")
    echo ""
    echo "────────────────────────────────────────────────────"
    echo "  $ds  dim=$dim  index=$index_dir"
    echo "────────────────────────────────────────────────────"
    $GRADLE -q runShardTest --args="--shards $index_dir --queries $embed_dir/queries.vec --docs $embed_dir --dim $dim"
done

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Done. Recall is in the table and SUMMARY lines above."
echo "═══════════════════════════════════════════════════════"
