"""
BPE training — skeleton for you to implement.

Pipeline:
    file -> chunks -> pre_tokenization -> word_counts -> BPE merges -> (vocab, merges)
"""

from __future__ import annotations

import os
from collections import Counter
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


def apply_merge_naive(
    word_counts: dict[tuple[bytes, ...], int],
    pair: tuple[bytes, bytes],
) -> dict[tuple[bytes, ...], int]:
    """
    Reference oracle: full-scan O(N) version. Kept for correctness testing
    against the incremental `apply_merge` below.

    Replace every occurrence of `pair` inside each word with pair[0] + pair[1].
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


def apply_merge(
    words: list[list[bytes]],
    word_freq: list[int],
    pair: tuple[bytes, bytes],
    pair_counts: dict[tuple[bytes, bytes], int],
    pair_to_words: dict[tuple[bytes, bytes], set[int]],
) -> None:
    """
    Incremental merge: only walks the words that actually contain `pair`,
    then subtract-then-add their pair contributions in pair_counts /
    pair_to_words. All three structures (words, pair_counts, pair_to_words)
    are mutated in place.
    """
    A, B = pair
    AB = A + B

    # snapshot — we mutate pair_to_words while iterating
    affected = list(pair_to_words[pair])

    for word_id in affected:
        word = words[word_id]
        freq = word_freq[word_id]

        # 1. subtract: group by distinct pair so a repeated pair (e.g. (a,b) in
        #    [a,b,a,b]) is touched exactly once — avoids double-discard / KeyError.
        old_pair_mult: Counter[tuple[bytes, bytes]] = Counter()
        for i in range(len(word) - 1):
            old_pair_mult[(word[i], word[i + 1])] += 1
        for old_pair, mult in old_pair_mult.items():
            pair_counts[old_pair] -= freq * mult
            if pair_counts[old_pair] == 0:
                del pair_counts[old_pair]
            pair_to_words[old_pair].discard(word_id)
            if not pair_to_words[old_pair]:
                del pair_to_words[old_pair]

        # 2. merge: replace [A, B] with [AB] in this word
        new_word: list[bytes] = []
        i = 0
        while i < len(word):
            if i + 1 < len(word) and word[i] == A and word[i + 1] == B:
                new_word.append(AB)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        words[word_id] = new_word

        # 3. add: same grouping trick for symmetry and speed
        new_pair_mult: Counter[tuple[bytes, bytes]] = Counter()
        for i in range(len(new_word) - 1):
            new_pair_mult[(new_word[i], new_word[i + 1])] += 1
        for new_pair, mult in new_pair_mult.items():
            pair_counts[new_pair] = pair_counts.get(new_pair, 0) + freq * mult
            pair_to_words.setdefault(new_pair, set()).add(word_id)


def build_index(
    words: list[list[bytes]],
    word_freq: list[int],
) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[int]]]:
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = {}

    for word_id, word in enumerate(words):
        freq = word_freq[word_id]
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + freq
            pair_to_words.setdefault(pair, set()).add(word_id)

    return pair_counts, pair_to_words


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

    # ---- Step 3: convert dict form -> list form (words must be mutable) ----
    words: list[list[bytes]] = [list(w) for w in word_counts]
    word_freq: list[int] = list(word_counts.values())

    # one-shot index build; later steps update it incrementally
    pair_counts, pair_to_words = build_index(words, word_freq)

    # ---- Step 4: BPE training loop ----
    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)

    for _ in range(num_merges):
        if not pair_counts:
            break

        # tie-break: lexicographically GREATER pair wins (GPT-2 convention)
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))

        apply_merge(words, word_freq, best, pair_counts, pair_to_words)

        merges.append(best)
        vocab[len(vocab)] = best[0] + best[1]

    return vocab, merges


if __name__ == "__main__":
    # Quick smoke test on a tiny input
    # vocab, merges = train_bpe("path/to/corpus.txt", vocab_size=300, special_tokens=["<|endoftext|>"])
    # print("first 10 merges:", merges[:10])
    pass
