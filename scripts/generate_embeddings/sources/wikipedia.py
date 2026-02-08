"""Load text from Wikipedia JSONL dump (e.g. simplewiki.json)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional


def load_wikipedia_jsonl(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    max_docs: Optional[int] = None,
    text_key: str = "text",
) -> Iterator[str]:
    """
    Yield raw document text from a JSONL file. Each line is one JSON object; we use text_key (default 'text').
    max_docs limits number of documents (lines) read.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Wikipedia dump not found: {path}")
    count = 0
    with open(path, "r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if text_key in data and data[text_key]:
                    yield data[text_key]
                    count += 1
                    if max_docs is not None and count >= max_docs:
                        return
            except (json.JSONDecodeError, KeyError):
                continue


def load_wikipedia_all(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    max_docs: Optional[int] = None,
    text_key: str = "text",
) -> list[str]:
    """Load all document texts from Wikipedia JSONL into a list."""
    return list(load_wikipedia_jsonl(path, encoding=encoding, max_docs=max_docs, text_key=text_key))
