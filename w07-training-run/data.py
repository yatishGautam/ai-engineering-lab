"""Data for W7: TinyShakespeare as a flat array of character ids.

The lesson's first stage, done honestly: tokenize ONCE, save a binary array, read random
crops from it. Nothing here is model-specific.

    python data.py        downloads, tokenizes, writes data/train.bin and data/val.bin

Shapes you will see:
    text        str          ~1.1 M characters
    ids         (N,)         uint16, one id per character (vocab is ~65, so uint8 would do;
                             uint16 keeps the same code working for a BPE vocab later)
    get_batch   x, y         both (B, T) int64 tensors; y is x shifted one position right
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
import torch

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = Path(__file__).parent / "data"      # gitignored: regenerate with `python data.py`


def download(path: Path = DATA_DIR / "input.txt") -> str:
    """Return the raw text, downloading it once."""
    # TODO(1): if `path` does not exist, mkdir its parent and urllib.request.urlretrieve(URL, path).
    #          Return path.read_text(encoding="utf-8").
    raise NotImplementedError


def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    """Character vocabulary: sorted unique chars -> ids, and the inverse."""
    # TODO(2): chars = sorted(set(text)); stoi maps char -> index; itos is the inverse.
    #          Expect len(chars) around 65 for TinyShakespeare.
    raise NotImplementedError


def encode(text: str, stoi: dict[str, int]) -> np.ndarray:
    """str -> np.uint16 array of ids."""
    # TODO(3): one id per character. np.array(..., dtype=np.uint16).
    raise NotImplementedError


def decode(ids, itos: dict[int, str]) -> str:
    """ids (list, array or 1-D tensor) -> str. Used to print generation samples."""
    # TODO(4): join itos[int(i)] for every id.
    raise NotImplementedError


def prepare() -> None:
    """Write data/train.bin (first 90%) and data/val.bin (last 10%) plus data/vocab.txt."""
    # TODO(5): text = download(); stoi, itos = build_vocab(text); ids = encode(text, stoi)
    #          n = int(0.9 * len(ids)); ids[:n].tofile(DATA_DIR / "train.bin"); ids[n:] likewise.
    #          Save the vocabulary too (one char per line, or json) so train.py can decode samples.
    #          Print vocab size, train tokens, val tokens.
    raise NotImplementedError


def load_split(split: str) -> np.ndarray:
    """Memory-map data/<split>.bin. Memmap, not read: the lesson's point about big data."""
    # TODO(6): np.memmap(DATA_DIR / f"{split}.bin", dtype=np.uint16, mode="r")
    raise NotImplementedError


def get_batch(data: np.ndarray, block_size: int, batch_size: int, device: str,
              fixed_offsets: np.ndarray | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Random crops of length block_size. x is the crop, y is the crop shifted by one.

    fixed_offsets: when given, use these start positions instead of random ones. train.py
    uses this for --overfit-one-batch (the same batch every step).
    """
    # TODO(7): ix = fixed_offsets if given else np.random.randint(0, len(data) - block_size, batch_size)
    #          x = stack of data[i : i + block_size] for i in ix        -> (B, T)
    #          y = stack of data[i + 1 : i + 1 + block_size] for i in ix -> (B, T)
    #          Convert with torch.from_numpy(np.array(..., dtype=np.int64)) and .to(device).
    raise NotImplementedError


if __name__ == "__main__":
    prepare()
