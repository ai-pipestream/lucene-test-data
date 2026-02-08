#!/bin/bash
# End-to-end: start DJL with BGE-M3, download data if needed, generate embeddings.
# Run from the lucene-test-data repo root.
#
# Usage:
#   ./run-embeddings.sh                  # defaults: bge_m3, 1024d, sentence, simplewiki
#   ./run-embeddings.sh --granularity paragraph
#   ./run-embeddings.sh --max-docs 5000  # quick test run
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Step 1: Ensure DJL Serving is running with BGE-M3 ==="
# Check if already running and model is READY
MODEL_STATUS=$(curl -sf http://localhost:8091/models 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for m in data.get('models', []):
        if m['modelName'] == 'bge_m3':
            print(m['status'])
            break
    else:
        print('NOT_FOUND')
except:
    print('ERROR')
" 2>/dev/null || echo "DOWN")

if [ "$MODEL_STATUS" = "READY" ]; then
    echo "DJL already running with bge_m3 READY. Skipping container restart."
else
    echo "Starting DJL Serving..."
    start-embedding-docker/restart-djl.sh
fi

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

echo ""
echo "=== Step 3: Install Python dependencies ==="
if pip install --quiet -r scripts/generate_embeddings/requirements.txt 2>/dev/null; then
    echo "  Dependencies installed."
else
    if python3 -c "import yaml, requests" 2>/dev/null; then
        echo "  pip install blocked (PEP 668), but pyyaml + requests already available."
    else
        echo "ERROR: Cannot install dependencies and they are not available." >&2
        echo "Create a venv: python3 -m venv .venv && source .venv/bin/activate" >&2
        exit 1
    fi
fi

echo ""
echo "=== Step 4: Generate embeddings ==="
python3 scripts/generate_embeddings/generate_embeddings.py \
    -c scripts/generate_embeddings/config.example.yaml \
    "$@"

echo ""
echo "=== Done! ==="
echo "Output files are in data/embeddings/"
ls -lh data/embeddings/ 2>/dev/null || echo "(no output files found)"
