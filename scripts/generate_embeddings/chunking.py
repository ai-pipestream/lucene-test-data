"""Split text into sentence or paragraph units for embedding."""
from __future__ import annotations

import re
from typing import List

# Sentence endings: same pattern as reference script (split on . ? ! with lookbehind)
SENTENCE_ENDINGS = re.compile(
    r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s"
)


def split_sentences(text: str, min_len: int = 10) -> List[str]:
    """Split into sentences: first by paragraphs (\\n\\n), then by sentence regex. Drops short fragments."""
    paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
    sentences = []
    for p in paras:
        sents = [s.strip() for s in SENTENCE_ENDINGS.split(p) if len(s.strip()) >= min_len]
        sentences.extend(sents)
    return sentences


def split_paragraphs(text: str, delimiter: str = "\n\n", min_len: int = 20) -> List[str]:
    """Split into paragraphs by delimiter. Drops very short chunks."""
    chunks = [p.strip() for p in text.split(delimiter) if len(p.strip()) >= min_len]
    return chunks


def chunk_text(
    text: str,
    granularity: str,
    *,
    paragraph_delimiter: str = "\n\n",
    min_sentence_len: int = 10,
    min_paragraph_len: int = 20,
) -> List[str]:
    """Return list of text chunks: either sentences or paragraphs."""
    if granularity == "sentence":
        return split_sentences(text, min_len=min_sentence_len)
    if granularity == "paragraph":
        return split_paragraphs(text, delimiter=paragraph_delimiter, min_len=min_paragraph_len)
    raise ValueError(f"granularity must be 'sentence' or 'paragraph', got {granularity!r}")
