# Test embeddings (small, committed)

Each dataset is written to a **named subdirectory** under this folder. All files for that run go in the subdir as `docs.vec`, `queries.vec`, `meta.json`.

| Name | Source | Contents |
|------|--------|----------|
| `unit-data-384-sentence` | test-texts (Alice, P&P) | sentence, 384d MiniLM |
| `unit-data-384-paragraph` | test-texts | paragraph, 384d MiniLM |
| `unit-data-1024-sentence` | test-texts | sentence, 1024d bge-m3 |
| `unit-data-1024-paragraph` | test-texts | paragraph, 1024d bge-m3 |
| `wiki-1024-sentences` | (future) | Wikipedia, 1024d sentences |
| … | | |

Generate with (from repo root):

```bash
# All four unit-data configs (folder method: Alice + Pride & Prejudice)
./scripts/generate_embeddings/run_test_text_embeddings.sh
```

Or run one config:

```bash
python scripts/generate_embeddings/generate_embeddings.py -c scripts/generate_embeddings/configs/text-sentence-bge-m3-1024.yaml
```

That writes to `embeddings/unit-data-1024-sentence/docs.vec`, `queries.vec`, `meta.json`.
