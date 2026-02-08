#!/bin/bash
# Register embedding models with DJL Serving.
#
# BGE-M3 auto-loads via mounted serving.properties at /opt/ml/model/bge-m3/.
# MiniLM is registered via the management API below.
set -e
BASE="${DJL_URL:-http://localhost:8091}"

echo "Registering models with DJL at $BASE..."

# MiniLM (384d) - registered via management API
echo "Registering all-MiniLM-L6-v2..."
curl -s -X POST "${BASE}/models?url=djl%3A%2F%2Fai.djl.huggingface.pytorch%2Fsentence-transformers%2Fall-MiniLM-L6-v2&model_name=all-MiniLM-L6-v2&synchronous=true"
echo ""

# BGE-M3 (1024d) - auto-loaded from /opt/ml/model/bge-m3/serving.properties
# If it hasn't auto-loaded yet, it will appear after the model download completes.
echo "BGE-M3 auto-loads via serving.properties (engine=Python, task=text-embedding)."
echo "If not yet ready, wait for model download to finish."

echo ""
echo "Check status: curl -s $BASE/models"
curl -s "$BASE/models"
echo ""
