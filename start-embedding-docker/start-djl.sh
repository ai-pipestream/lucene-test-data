#!/bin/bash

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

echo "Starting DJL with image: $DJL_IMAGE"

# Ensure we are in the script's directory
cd "$(dirname "$0")"

docker compose up -d
