#!/bin/bash
# Restart DJL without BGE-M3 (no auto-load). Then load MiniLM only for 384d embeddings.
# Use this when 1024d is already done and you want to run wiki-384-* without GPU OOM.

set -e

if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
    echo "GPU detected."
    export DJL_IMAGE="deepjavalibrary/djl-serving:0.36.0-pytorch-gpu"
    export DOCKER_DRIVER="nvidia"
    export GPU_COUNT="all"
else
    echo "No GPU detected. Falling back to CPU."
    export DJL_IMAGE="deepjavalibrary/djl-serving:0.36.0-pytorch-cpu"
    export DOCKER_DRIVER="none"
    export GPU_COUNT="0"
fi

cd "$(dirname "$0")"

echo "Stopping DJL..."
docker compose down

echo "Starting DJL without BGE-M3 (MiniLM-only mode)..."
docker compose -f docker-compose.yml -f docker-compose.minilm-only.yml up -d

echo "Waiting for DJL server..."
for i in $(seq 1 120); do
    if curl -sf http://localhost:8091/models >/dev/null 2>&1; then
        echo "Server is up."
        break
    fi
    sleep 2
done

if ! curl -sf http://localhost:8091/models >/dev/null 2>&1; then
    echo "DJL server did not start in time. Check: docker compose -f docker-compose.yml -f docker-compose.minilm-only.yml logs"
    exit 1
fi

echo "Loading MiniLM (all-MiniLM-L6-v2)..."
./load-models.sh

echo "Waiting for all-MiniLM-L6-v2 READY..."
for i in $(seq 1 120); do
    STATUS=$(curl -sf http://localhost:8091/models 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for m in data.get('models', []):
        if m['modelName'] == 'all-MiniLM-L6-v2':
            print(m['status'])
            break
    else:
        print('NOT_FOUND')
except:
    print('ERROR')
" 2>/dev/null)
    if [ "$STATUS" = "READY" ]; then
        echo "all-MiniLM-L6-v2 is READY."
        curl -s http://localhost:8091/models
        echo ""
        exit 0
    fi
    if [ "$((i % 15))" -eq 0 ]; then
        echo "  Still loading... (${i}s, status=$STATUS)"
    fi
    sleep 2
done

echo "MiniLM did not become READY in time. Check: docker compose logs"
exit 1
