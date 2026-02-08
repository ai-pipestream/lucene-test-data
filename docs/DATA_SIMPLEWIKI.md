# Simple Wikipedia data (simplewiki.json)

The embedding pipeline expects a JSONL file with one JSON object per line, each with a `"text"` key (article body). The file is **not** committed; it’s in `.gitignore`.

## Option 1: Copy an existing dump

If you already have a Simple Wikipedia JSONL dump (e.g. from another project):

```bash
cp /path/to/simplewiki.json lucene-test-data/data/simplewiki.json
```

Run the embedding script from the **lucene-test-data** repo root so that `wikipedia.path: data/simplewiki.json` in config resolves correctly.

## Option 2: Download via script (Hugging Face)

From the repo root:

```bash
python scripts/generate_embeddings/fetch_simplewiki.py
```

This checks for `data/simplewiki.json`. If it’s missing, it downloads Simple English Wikipedia from Hugging Face (`wikipedia` dataset, config `20220301.simple`) and writes JSONL to `data/simplewiki.json`. The result is ~205k articles (~130–230 MB). For a larger dump (e.g. full simplewiki), use Option 3 or copy an existing file (Option 1).

Requires: `pip install datasets` (or use the same env as the embedding script).

## Option 3: Build from official dump (full size)

For the full Simple English Wikipedia (~2.9 GB JSONL, 550k+ articles):

1. Download the XML dump from [dumps.wikimedia.org/simplewiki/](https://dumps.wikimedia.org/simplewiki/) (e.g. `simplewiki-YYYYMMDD-pages-articles.xml.bz2`).
2. Convert to JSONL using [Gensim’s segment_wiki](https://radimrehurek.com/gensim/scripts/segment_wiki.html), or another tool that outputs one JSON per line with article text.
3. Save as `data/simplewiki.json` (or set `wikipedia.path` in config to your file path).

## Config

Default path in `scripts/generate_embeddings/config.example.yaml`:

```yaml
wikipedia:
  path: data/simplewiki.json
```

Paths are relative to the **current working directory** when you run `generate_embeddings.py`. Run from **lucene-test-data** repo root so that `data/simplewiki.json` is found. Or use an absolute path or `--wikipedia-path` on the CLI.
