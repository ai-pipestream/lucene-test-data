#!/usr/bin/env bash
# Generate test embeddings via the folder method (text_dir = Alice + Pride & Prejudice).
# Run from lucene-test-data repo root. Requires DJL serving with models loaded.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

echo "Generating test-text embeddings (folder method: test-texts/)..."
for config in text-sentence-minilm-384 text-paragraph-minilm-384 text-sentence-bge-m3-1024 text-paragraph-bge-m3-1024; do
  echo "  Running $config..."
  python scripts/generate_embeddings/generate_embeddings.py -c "scripts/generate_embeddings/configs/${config}.yaml"
done
echo "Done. Output in index-builder/src/main/resources/embeddings/"
