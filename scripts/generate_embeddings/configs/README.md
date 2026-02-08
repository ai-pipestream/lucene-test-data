# Embedding configs (four sets)

| Config | Output dir (name) | Dimension | Granularity | Model |
|--------|-------------------|-----------|-------------|--------|
| `text-sentence-minilm-384.yaml` | `unit-data-384-sentence` | 384 | sentence | all-MiniLM-L6-v2 |
| `text-paragraph-minilm-384.yaml` | `unit-data-384-paragraph` | 384 | paragraph | all-MiniLM-L6-v2 |
| `text-sentence-bge-m3-1024.yaml` | `unit-data-1024-sentence` | 1024 | sentence | bge_m3 |
| `text-paragraph-bge-m3-1024.yaml` | `unit-data-1024-paragraph` | 1024 | paragraph | bge_m3 |

All use **text_dir** → `index-builder/src/main/resources/test-texts` (Alice + Pride & Prejudice). Each run writes to `embeddings/<name>/` with `docs.vec`, `queries.vec`, `meta.json`. For Wikipedia you’d use names like `wiki-1024-sentences`.

Run from **lucene-test-data** repo root:

```bash
python scripts/generate_embeddings/generate_embeddings.py -c scripts/generate_embeddings/configs/<config>.yaml
```
