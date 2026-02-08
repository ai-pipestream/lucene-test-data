#!/bin/bash
# Build all Lucene indices from pre-sharded embedding datasets.
#
# For each of the 4 datasets, builds 16 index shards (16 threads), then
# merges down to 8, 4, and 2 shards using Lucene addIndexes (segment copy).
#
# Prerequisites:
#   - Embedding datasets generated: ./run-all-embeddings.sh
#   - Each dataset has 16 vec shard files (docs-shard-{0..15}.vec)
#
# Usage:
#   ./run-all-indices.sh                    # build all
#   ./run-all-indices.sh --threads 8        # limit to 8 build threads
#   ./run-all-indices.sh --skip-build       # only run merges (if 16-shard indices exist)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

GRADLE="index-builder/gradlew -p index-builder"
EMBEDDINGS_BASE="$SCRIPT_DIR/data/embeddings"
INDICES_BASE="$SCRIPT_DIR/data/indices"
THREADS=16
BATCH_SIZE=1000
SKIP_BUILD=false

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --threads|-t)  THREADS="$2"; shift 2 ;;
        --batch-size)  BATCH_SIZE="$2"; shift 2 ;;
        --skip-build)  SKIP_BUILD=true; shift ;;
        --help|-h)
            echo "Usage: $0 [--threads N] [--batch-size N] [--skip-build]"
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
SHARD_COUNTS=(2 4 8 16)

# ── Step 1: Build 16-shard indices from pre-sharded vec files ────────
echo "═══════════════════════════════════════════════════════"
echo "  Index Builder: 4 datasets × 4 shard counts (2, 4, 8, 16)"
echo "  Threads: $THREADS  Batch size: $BATCH_SIZE"
echo "═══════════════════════════════════════════════════════"

BUILT=0
MERGED=0
SKIPPED=0
FAILED=0

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "────────────────────────────────────────────────────"
    echo "Dataset: $ds"
    echo "────────────────────────────────────────────────────"

    ds_embed="$EMBEDDINGS_BASE/$ds"
    if [ ! -f "$ds_embed/meta.json" ]; then
        echo "  SKIP: No embeddings found at $ds_embed"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    ds_index="$INDICES_BASE/$ds"
    shards16_dir="$ds_index/shards-16"

    # Build 16 shards (the finest granularity)
    if [ "$SKIP_BUILD" = true ]; then
        echo "  --skip-build: skipping 16-shard build"
    elif [ -d "$shards16_dir/shard-15" ]; then
        echo "  16-shard index already exists. Skipping build."
    else
        echo "  Building 16 index shards ($THREADS threads)..."
        if $GRADLE -q run --args="--dataset $ds_embed --output $shards16_dir --num-shards 16 --threads $THREADS --batch-size $BATCH_SIZE"; then
            BUILT=$((BUILT + 1))
            echo "  Done: $shards16_dir"
        else
            echo "  ERROR: Build failed for $ds"
            FAILED=$((FAILED + 1))
            continue
        fi
    fi

    # Verify 16-shard dir exists before merging
    if [ ! -d "$shards16_dir/shard-0" ]; then
        echo "  ERROR: 16-shard index not found at $shards16_dir. Cannot merge."
        FAILED=$((FAILED + 1))
        continue
    fi

    # Merge down to 8, 4, 2 shards
    for target in 8 4 2; do
        target_dir="$ds_index/shards-$target"
        if [ -d "$target_dir/shard-$((target - 1))" ]; then
            echo "  ${target}-shard index already exists. Skipping merge."
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        echo "  Merging 16 → $target shards..."
        merge_threads=$((THREADS < target ? THREADS : target))
        if $GRADLE -q mergeShards --args="--input $shards16_dir --output $target_dir --source-shards 16 --target-shards $target --threads $merge_threads"; then
            MERGED=$((MERGED + 1))
            echo "  Done: $target_dir"
        else
            echo "  ERROR: Merge failed for $ds shards-$target"
            FAILED=$((FAILED + 1))
        fi
    done
done

# ── Summary ───────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  All done!  Built: $BUILT  Merged: $MERGED  Skipped: $SKIPPED  Failed: $FAILED"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Output:"
for ds in "${DATASETS[@]}"; do
    for n in "${SHARD_COUNTS[@]}"; do
        dir="$INDICES_BASE/$ds/shards-$n"
        if [ -d "$dir/shard-0" ]; then
            SIZE=$(du -sh "$dir" | cut -f1)
            echo "  $dir/  ($SIZE, $n shards)"
        else
            echo "  $dir/  (missing)"
        fi
    done
done
