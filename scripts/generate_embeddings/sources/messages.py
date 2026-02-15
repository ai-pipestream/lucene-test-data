"""Load text messages from a file (one message per line)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def load_messages_all(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    max_docs: Optional[int] = None,
) -> list[str]:
    """
    Load messages from a text file. Each non-empty line is one message.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Messages file not found: {path}")

    messages = []
    with open(path, encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(line)
                if max_docs is not None and len(messages) >= max_docs:
                    break
    return messages
