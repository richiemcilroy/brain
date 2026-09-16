"""Pinned literary holdout text, outside the two adjustment corpora.

The text is the plain UTF-8 Project Gutenberg eBook #11. Strip only its
Gutenberg header/license footer using explicit START/END marker lines.
The raw file and resulting book body have fixed hashes; neither is committed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


PATH = Path("/tmp/human_brain_alice_pg11.txt")
URL = "https://www.gutenberg.org/cache/epub/11/pg11.txt"
BOOK_PAGE = "https://www.gutenberg.org/ebooks/11"
RAW_SHA256 = "01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e"
BODY_SHA256 = "4e04ea77acf3b0215cae2089c977bc07526fc834597a1d463598d895354ba41d"
START = "*** START OF THE PROJECT GUTENBERG EBOOK"
END = "*** END OF THE PROJECT GUTENBERG EBOOK"


def alice_body() -> str:
    if not PATH.is_file():
        raise FileNotFoundError(f"pinned literary holdout missing: {PATH}; {URL}")
    raw = PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != RAW_SHA256:
        raise ValueError("literary holdout raw text differs from pinned bytes")
    # Match Python's universal-newline text reading used to pin the body.
    text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    first_marker = text.index(START)
    first = text.index("\n", first_marker) + 1
    last = text.index(END, first)
    body = text[first:last].strip()
    if hashlib.sha256(body.encode("utf-8")).hexdigest() != BODY_SHA256:
        raise ValueError("literary holdout body differs after marker removal")
    return body
