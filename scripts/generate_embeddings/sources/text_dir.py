"""Load text from a directory of plain-text files."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional


def load_text_dir(
    path: str | Path,
    *,
    glob: str = "**/*.txt",
    encoding: str = "utf-8",
    max_docs: Optional[int] = None,
) -> Iterator[str]:
    """
    Yield raw file contents from directory. Each file is one "document".
    max_docs limits number of files read.
    """
    path = Path(path)
    if not path.is_dir():
        raise NotADirectoryError(f"Text dir not found or not a directory: {path}")
    count = 0
    for fp in sorted(path.glob(glob)):
        if not fp.is_file():
            continue
        try:
            text = fp.read_text(encoding=encoding)
            if text.strip():
                yield text
                count += 1
                if max_docs is not None and count >= max_docs:
                    return
        except (OSError, UnicodeDecodeError):
            continue


def load_text_dir_all(
    path: str | Path,
    *,
    glob: str = "**/*.txt",
    encoding: str = "utf-8",
    max_docs: Optional[int] = None,
) -> list[str]:
    """Load all document texts from directory into a list."""
    return list(load_text_dir(path, glob=glob, encoding=encoding, max_docs=max_docs))
