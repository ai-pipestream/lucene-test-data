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


def split_paragraphs(
    text: str,
    delimiter: str = "\n\n",
    min_len: int = 20,
    sentences_per_paragraph: int = 3,
    max_chars: int = 1024,
) -> List[str]:
    """Split into paragraphs by delimiter, falling back to sentence grouping.

    If delimiter-based splitting yields only 1 chunk (e.g. flat text with no
    newlines), groups every ``sentences_per_paragraph`` sentences into a chunk.
    Chunks exceeding ``max_chars`` are split in half by sentence boundary.
    """
    chunks = [p.strip() for p in text.split(delimiter) if len(p.strip()) >= min_len]
    if len(chunks) > 1:
        return chunks

    # Fallback: group sentences into paragraph-sized chunks
    sentences = split_sentences(text, min_len=10)
    if not sentences:
        return chunks  # return the single chunk if we can't split further

    grouped: List[str] = []
    for i in range(0, len(sentences), sentences_per_paragraph):
        chunk = " ".join(sentences[i : i + sentences_per_paragraph])
        if len(chunk) > max_chars:
            sents = sentences[i : i + sentences_per_paragraph]
            if len(sents) > 1:
                # Split in half by sentence boundary
                mid = len(sents) // 2
                grouped.append(" ".join(sents[:mid]))
                grouped.append(" ".join(sents[mid:]))
            else:
                grouped.append(chunk)
        else:
            grouped.append(chunk)

    # Enforce max_chars on all chunks — split oversized ones on word boundaries
    result: List[str] = []
    for chunk in grouped:
        if len(chunk) <= max_chars:
            result.append(chunk)
        else:
            s = chunk
            while len(s) > max_chars:
                cut = s.rfind(" ", 0, max_chars)
                if cut <= 0:
                    cut = max_chars
                result.append(s[:cut])
                s = s[cut:].lstrip()
            if s:
                result.append(s)

    return [c for c in result if len(c) >= min_len]


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
