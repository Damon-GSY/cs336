"""
BPE training — skeleton for you to implement.

Pipeline:
    file -> chunks -> pre_tokenization -> word_counts -> BPE merges -> (vocab, merges)
"""

from __future__ import annotations

import os
from typing import BinaryIO

import regex as re

from pretokenization_example import find_chunk_boundaries, pre_tokenization


def count_pairs(
    word_counts: dict[tuple[bytes, ...], int],
) -> dict[tuple[bytes, bytes], int]:
    """
    Count adjacent byte-pair frequencies across all words, weighted by word count.

    Input:
        word_counts: {(b'l', b'o', b'w'): 5, ...}
    Output:
        pair_counts: {(b'l', b'o'): 5, (b'o', b'w'): 5, ...}
    """
    pair_counts: dict[tuple[bytes, bytes], int] = {}

    for word, count in word_counts.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + count

    return pair_counts



def apply_merge(
    word_counts: dict[tuple[bytes, ...], int],
    pair: tuple[bytes, bytes],
) -> dict[tuple[bytes, ...], int]:
    """
    Replace every occurrence of `pair` inside each word with the merged element pair[0] + pair[1].

    Returns a NEW dict.
    """
    merged = pair[0] + pair[1]
    new_word_counts: dict[tuple[bytes, ...], int] = {}

    for word, count in word_counts.items():
        new_word = []
        i = 0

        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        new_word = tuple(new_word)
        new_word_counts[new_word] = new_word_counts.get(new_word, 0) + count

    return new_word_counts




def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 4,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.

    Returns:
        vocab: dict[int, bytes]              token_id -> bytes
        merges: list[tuple[bytes, bytes]]    ordered merge history
    """
    # ---- Step 1: build initial vocab (256 bytes + special tokens) ----
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i in range(len(special_tokens)):
        vocab[i + 256] = special_tokens[i].encode("utf-8")

    # ---- Step 2: pre-tokenize file -> word_counts ----
    word_counts: dict[tuple[bytes, ...], int] = {}
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            local = pre_tokenization(chunk, special_tokens)
            for key, cnt in local.items():
                word_counts[key] = word_counts.get(key, 0) + cnt
    # ---- Step 3: BPE training loop ----
    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)

    for _ in range(num_merges):
        # 3a. count pair frequencies
        pair_counts = count_pairs(word_counts)
        if not pair_counts:
            break

        # 3b. pick the best pair
        #     tie-break: lexicographically GREATER pair wins (GPT-2 convention)
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))

        # 3c. apply merge to word_counts
        word_counts = apply_merge(word_counts, best)

        # 3d. record merge + add new token to vocab
        merges.append(best)
        vocab[len(vocab)] = best[0] + best[1]

    return vocab, merges


if __name__ == "__main__":
    # Quick smoke test on a tiny input
    # vocab, merges = train_bpe("path/to/corpus.txt", vocab_size=300, special_tokens=["<|endoftext|>"])
    # print("first 10 merges:", merges[:10])
    pass
