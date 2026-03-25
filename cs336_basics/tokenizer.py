from __future__ import annotations

import os
import re
from collections.abc import Iterable, Iterator

import regex

GPT2_PATTERN = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens) if special_tokens is not None else []

        self.bytes_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }

        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped = [re.escape(token) for token in sorted_tokens]
            self.special_pattern = re.compile("(" + "|".join(escaped) + ")")
        else:
            self.special_pattern = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        raise NotImplementedError

    def _encode_chunk(self, text: str) -> list[int]:
        if not text:
            return []

        result: list[int] = []
        for match in GPT2_PATTERN.finditer(text):
            pre_token = match.group(0)
            tokens: list[bytes] = [bytes([b]) for b in pre_token.encode("utf-8")]

            while len(tokens) >= 2:
                best_pair = None
                best_rank = float("inf")

                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    if pair in self.merge_ranks:
                        rank = self.merge_ranks[pair]
                        if rank < best_rank:
                            best_rank = rank
                            best_pair = pair

                if best_pair is None:
                    break

                new_tokens: list[bytes] = []
                i = 0
                while i < len(tokens):
                    if (
                        i < len(tokens) - 1
                        and tokens[i] == best_pair[0]
                        and tokens[i + 1] == best_pair[1]
                    ):
                        new_tokens.append(best_pair[0] + best_pair[1])
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens

            result.extend(self.bytes_to_id[token] for token in tokens)

        return result

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        if self.special_pattern is None:
            return self._encode_chunk(text)

        parts = self.special_pattern.split(text)
        ids: list[int] = []

        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                part_bytes = part.encode("utf-8")
                if part_bytes in self.bytes_to_id:
                    ids.append(self.bytes_to_id[part_bytes])
            else:
                ids.extend(self._encode_chunk(part))

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            if self.special_pattern is not None:
                parts = self.special_pattern.split(buffer)
                if len(parts) > 1:
                    for part in parts[:-1]:
                        if part:
                            if part in self.special_tokens:
                                part_bytes = part.encode("utf-8")
                                if part_bytes in self.bytes_to_id:
                                    yield self.bytes_to_id[part_bytes]
                            else:
                                yield from self._encode_chunk(part)
                    buffer = parts[-1]
        if buffer:
            if self.special_pattern is not None:
                parts = self.special_pattern.split(buffer)
                for part in parts:
                    if part:
                        if part in self.special_tokens:
                            part_bytes = part.encode("utf-8")
                            if part_bytes in self.bytes_to_id:
                                yield self.bytes_to_id[part_bytes]
                        else:
                            yield from self._encode_chunk(part)
            else:
                yield from self._encode_chunk(buffer)

    def decode(self, ids: list[int]) -> str:
        byte_pieces: list[bytes] = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_pieces.append(self.vocab[token_id])
        return b"".join(byte_pieces).decode("utf-8", errors="replace")


def _get_pair_counts(
    word_freqs: dict[tuple[bytes, ...], int],
) -> dict[tuple[bytes, bytes], int]:
    counts: dict[tuple[bytes, bytes], int] = {}
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            counts[pair] = counts.get(pair, 0) + freq
    return counts


def _merge_pair(
    word_freqs: dict[tuple[bytes, ...], int],
    pair: tuple[bytes, bytes],
) -> dict[tuple[bytes, ...], int]:
    new_word_freqs: dict[tuple[bytes, ...], int] = {}
    merged = pair[0] + pair[1]

    for word, freq in word_freqs.items():
        new_word: list[bytes] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word_freqs[tuple(new_word)] = freq

    return new_word_freqs


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    special_pattern = None
    if special_tokens:
        sorted_tokens = sorted(special_tokens, key=len, reverse=True)
        escaped = [re.escape(t) for t in sorted_tokens]
        special_pattern = re.compile("(" + "|".join(escaped) + ")")

    chunks: list[str] = []
    if special_pattern:
        parts = special_pattern.split(text)
        chunks = [p for p in parts if p and p not in special_tokens]
    else:
        chunks = [text] if text else []

    word_freqs: dict[tuple[bytes, ...], int] = {}
    for chunk in chunks:
        for match in GPT2_PATTERN.finditer(chunk):
            pre_token = match.group(0)
            word = tuple(bytes([b]) for b in pre_token.encode("utf-8"))
            word_freqs[word] = word_freqs.get(word, 0) + 1

    vocab: dict[int, bytes] = {}
    for i in range(256):
        vocab[i] = bytes([i])

    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in vocab.values():
            vocab[len(vocab)] = token_bytes

    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        pair_counts = _get_pair_counts(word_freqs)
        if not pair_counts:
            break

        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]

        word_freqs = _merge_pair(word_freqs, best_pair)
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

    return vocab, merges
