"""Pinned WikiText-2 raw splits and deterministic disjoint token windows.

The archive is a pinned ggml-org CI WikiText-2 raw artifact. Its member names
follow the Salesforce dataset script's train/valid/test layout. Text stays
outside the repository; callers record member and token-ID hashes so the exact
data used in a run can be checked without committing the corpus.
"""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import numpy as np


ZIP = Path("/tmp/human_brain_wikitext2_raw.zip")
ZIP_SHA256 = "ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11"
DATASET_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/927b3642933080f1b0e811e2f916e14c292992f9/wikitext-2-raw-v1.zip"
DATASET_CARD = "https://huggingface.co/datasets/Salesforce/wikitext"
MEMBERS = {
    "train": ("wikitext-2-raw/wiki.train.raw",
              "6707892fa3788b5ab9ed78ab5ff37d9fe825f6011a2ad4fcd6a6d467f0e7da57"),
    "valid": ("wikitext-2-raw/wiki.valid.raw",
              "4cd0f6876d07a413aa911261ff6d363c72d757d47f0fdd6015702014c89cb9c7"),
    "test": ("wikitext-2-raw/wiki.test.raw",
             "173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08"),
}


def wiki_raw(split: str) -> str:
    if split not in MEMBERS:
        raise ValueError(f"unknown WikiText split: {split}")
    if not ZIP.is_file() or hashlib.sha256(ZIP.read_bytes()).hexdigest() != ZIP_SHA256:
        raise FileNotFoundError(f"pinned WikiText archive differs: {ZIP}; {DATASET_URL}")
    member, digest = MEMBERS[split]
    with zipfile.ZipFile(ZIP) as archive:
        raw = archive.read(member)
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"pinned WikiText {split} member differs")
    return raw.decode("utf-8")


def spaced_starts(total: int, *, region_end: int, count: int,
                  window: int) -> list[int]:
    """Evenly spread nonoverlapping [input + label] windows in a region."""
    if not (0 < region_end <= total and count > 0 and window > 0):
        raise ValueError("invalid region or window counts")
    spacing = region_end // count
    starts = [int((index + 0.5) * spacing) for index in range(count)]
    if (spacing < window or starts[-1] + window > region_end or
            any(left + window > right for left, right in zip(
                starts, starts[1:]))):
        raise ValueError("evenly spaced windows overlap or leave the train region")
    return starts


def percentage_starts(total: int, percentages: tuple[int, ...],
                      window: int) -> dict[str, int]:
    if not percentages or window < 1:
        raise ValueError("fixed percentages and positive window required")
    starts = {str(pct): int(total * pct / 100) for pct in percentages}
    ordered = sorted(starts.values())
    if (len(starts) != len(percentages) or
            min(percentages) < 0 or max(percentages) >= 100 or
            ordered[-1] + window > total or
            any(left + window > right for left, right in zip(
                ordered, ordered[1:]))):
        raise ValueError("fixed windows overlap or exceed text")
    return starts


def window_rows(ids: np.ndarray, starts: dict[str, int] | list[int],
                window: int) -> dict[str, dict]:
    items = (starts.items() if isinstance(starts, dict)
             else ((str(index), start) for index, start in enumerate(starts)))
    return {
        key: {
            "token_window": [int(start), int(start + window)],
            "token_ids_sha256": hashlib.sha256(
                ids[start:start + window].tobytes()).hexdigest(),
        }
        for key, start in items
    }
