#!/bin/bash
# Restart DJL Serving container (stop, remove, start fresh).
# Waits for the server AND bge_m3 model to be READY.

set -e

# Detect GPU availability
if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
    echo "GPU detected using nvidia-smi."
    export DJL_IMAGE="deepjavalibrary/djl-serving:0.36.0-pytorch-gpu"
    export DOCKER_DRIVER="nvidia"
    export GPU_COUNT="all"
else
    echo "No GPU detected or nvidia-smi failed. Falling back to CPU."
    export DJL_IMAGE="deepjavalibrary/djl-serving:0.36.0-pytorch-cpu"
    export DOCKER_DRIVER="none"
    export GPU_COUNT="0"
fi

echo "Restarting DJL with image: $DJL_IMAGE"

# Ensure we are in the script's directory
cd "$(dirname "$0")"

docker compose down
docker compose up -d

echo "Waiting for DJL server to start..."
for i in $(seq 1 120); do
    if curl -sf http://localhost:8091/models >/dev/null 2>&1; then
        break
    fi
    sleep 2
done

if ! curl -sf http://localhost:8091/models >/dev/null 2>&1; then
    echo "DJL server did not start within 240 seconds. Check logs: docker compose logs"
    exit 1
fi

echo "Server is up. Waiting for bge_m3 model to be READY..."
for i in $(seq 1 120); do
    STATUS=$(curl -sf http://localhost:8091/models 2>/dev/null | python3 -c "
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
" 2>/dev/null)
    if [ "$STATUS" = "READY" ]; then
        echo "bge_m3 model is READY!"
        curl -s http://localhost:8091/models
        echo ""
        exit 0
    fi
    if [ "$((i % 15))" -eq 0 ]; then
        echo "  Still loading... (${i}s, status=$STATUS)"
    fi
    sleep 2
done

echo "bge_m3 model did not become READY within 240 seconds. Check logs: docker compose logs"
exit 1
