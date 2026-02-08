#!/bin/bash
# Generate all 4 embedding sets (2 models x 2 granularities).
# Skips sets that already have output files. Idempotent and resumable.
#
# Run from the lucene-test-data repo root:
#   ./run-all-embeddings.sh
#   ./run-all-embeddings.sh --max-docs 5000   # quick test with fewer docs
#   ./run-all-embeddings.sh --batch-size 1000  # crank up batch size
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

EMBED_SCRIPT="scripts/generate_embeddings/generate_embeddings.py"
CONFIG="scripts/generate_embeddings/config.example.yaml"
OUTPUT_BASE="data/embeddings"
EXTRA_ARGS=("$@")

# ── The 4 embedding sets ─────────────────────────────────────────────
#   output-name            model                  dim   granularity  batch_size
#   BGE-M3 (~568M params) → batch 500 on 16GB VRAM; MiniLM (~22M) → batch 1000
SETS=(
  "wiki-1024-sentences    bge_m3                 1024  sentence     500"
  "wiki-1024-paragraphs   bge_m3                 1024  paragraph    500"
  "wiki-384-sentences     all-MiniLM-L6-v2       384   sentence     1000"
  "wiki-384-paragraphs    all-MiniLM-L6-v2       384   paragraph    1000"
)

# ── Helper: check if a set is already complete ────────────────────────
set_is_complete() {
    local name="$1"
    local dir="$OUTPUT_BASE/$name"
    [ -f "$dir/docs.vec" ] && [ -f "$dir/queries.vec" ] && [ -f "$dir/meta.json" ]
}

# ── Helper: check if a DJL model is READY ────────────────────────────
model_is_ready() {
    local model_name="$1"
    local status
    status=$(curl -sf http://localhost:8091/models 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for m in data.get('models', []):
        if m['modelName'] == '$model_name':
            print(m['status'])
            break
    else:
        print('NOT_FOUND')
except:
    print('ERROR')
" 2>/dev/null || echo "DOWN")
    [ "$status" = "READY" ]
}

# ── Step 1: Ensure DJL is running with BGE-M3 ────────────────────────
echo "=== Step 1: Ensure DJL Serving is running ==="
if model_is_ready "bge_m3"; then
    echo "DJL already running with bge_m3 READY."
else
    echo "Starting DJL Serving..."
    start-embedding-docker/restart-djl.sh
fi

# ── Step 2: Ensure simplewiki data exists ─────────────────────────────
echo ""
echo "=== Step 2: Ensure simplewiki data exists ==="
if [ ! -f data/simplewiki.json ]; then
    echo "Downloading Simple English Wikipedia..."
    pip install --quiet datasets
    python3 scripts/generate_embeddings/fetch_simplewiki.py
else
    SIZE=$(du -sh data/simplewiki.json | cut -f1)
    echo "data/simplewiki.json already exists ($SIZE). Skipping download."
fi

# ── Step 3: Install Python dependencies ───────────────────────────────
echo ""
echo "=== Step 3: Install Python dependencies ==="
if pip install --quiet -r scripts/generate_embeddings/requirements.txt 2>/dev/null; then
    echo "  Dependencies installed."
else
    # PEP 668: externally-managed-environment. Check if already importable.
    if python3 -c "import yaml, requests" 2>/dev/null; then
        echo "  pip install blocked (PEP 668), but pyyaml + requests already available."
    else
        echo "ERROR: Cannot install dependencies and they are not available." >&2
        echo "Create a venv: python3 -m venv .venv && source .venv/bin/activate" >&2
        exit 1
    fi
fi

# ── Step 4: Run each embedding set ────────────────────────────────────
echo ""
echo "=== Step 4: Generate all 4 embedding sets ==="

COMPLETED=0
SKIPPED=0
FAILED=0

for set_line in "${SETS[@]}"; do
    read -r name model dim granularity default_batch <<< "$set_line"

    echo ""
    echo "────────────────────────────────────────────────────"
    echo "Set: $name (model=$model, dim=$dim, granularity=$granularity, batch=$default_batch)"
    echo "────────────────────────────────────────────────────"

    if set_is_complete "$name"; then
        echo "  Already complete. Skipping."
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Ensure the model is loaded
    if ! model_is_ready "$model"; then
        if [ "$model" = "bge_m3" ]; then
            echo "  bge_m3 not ready -- restarting DJL..."
            start-embedding-docker/restart-djl.sh
        else
            echo "  Loading $model via management API..."
            start-embedding-docker/load-models.sh
            # Wait for it
            echo "  Waiting for $model to be READY..."
            for i in $(seq 1 60); do
                if model_is_ready "$model"; then break; fi
                sleep 2
            done
            if ! model_is_ready "$model"; then
                echo "  ERROR: $model did not become READY. Skipping set."
                FAILED=$((FAILED + 1))
                continue
            fi
        fi
    fi

    echo "  Running embeddings..."
    if python3 "$EMBED_SCRIPT" -c "$CONFIG" \
        --model-name "$model" --dim "$dim" \
        --granularity "$granularity" \
        --output-name "$name" \
        --batch-size "$default_batch" \
        "${EXTRA_ARGS[@]}"; then
        COMPLETED=$((COMPLETED + 1))
    else
        echo "  ERROR: Embedding generation failed for $name"
        FAILED=$((FAILED + 1))
    fi
done

# ── Summary ───────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  All done! Completed: $COMPLETED  Skipped: $SKIPPED  Failed: $FAILED"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Output:"
for set_line in "${SETS[@]}"; do
    read -r name _ _ _ <<< "$set_line"
    dir="$OUTPUT_BASE/$name"
    if [ -f "$dir/docs.vec" ]; then
        SIZE=$(du -sh "$dir/docs.vec" | cut -f1)
        echo "  $dir/  ($SIZE)"
    else
        echo "  $dir/  (missing)"
    fi
done
