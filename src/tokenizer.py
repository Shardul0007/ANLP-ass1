from __future__ import annotations

import os
import json
import collections
from typing import Optional, List, Dict, Tuple, Set, Any

# Special tokens
BPE_PAD = "<pad>"
BPE_BOS = "<bos>"
BPE_EOS = "<eos>"
BPE_UNK = "<unk>"
DEFAULT_SPECIAL_TOKENS = [BPE_PAD, BPE_BOS, BPE_EOS, BPE_UNK]


class EncodedOutput:
    """Wrapper holding encoded token IDs and tokens."""

    def __init__(self, ids: List[int], tokens: List[str]):
        self.ids = ids
        self.tokens = tokens


class BPETokenizer:
    """Fast Byte-Pair Encoding (BPE) Tokenizer implemented from scratch.

    Zero third-party library dependencies (no tokenizers, tiktoken, or sentencepiece).
    Learns subword merge rules from character/byte representations up to vocab_size.
    """

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        merges: Optional[List[Tuple[str, str]]] = None,
        special_tokens: Optional[List[str]] = None,
        is_cipher: bool = False,
    ):
        self.special_tokens = special_tokens or DEFAULT_SPECIAL_TOKENS
        self.pad_token = BPE_PAD
        self.bos_token = BPE_BOS
        self.eos_token = BPE_EOS
        self.unk_token = BPE_UNK
        self.is_cipher = is_cipher

        self.vocab: Dict[str, int] = vocab or {}
        self.inv_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.merges: List[Tuple[str, str]] = merges or []
        self.bpe_ranks: Dict[Tuple[str, str], int] = {
            tuple(pair): i for i, pair in enumerate(self.merges)
        }
        self.cache: Dict[str, List[str]] = {}

    @classmethod
    def _pre_tokenize(cls, text: str, is_cipher: bool) -> List[str]:
        if is_cipher:
            # Cipher strings are delimited by '|' every 8 bits (e.g. "01100001|01100010|")
            blocks = text.strip().split("|")
            return [b + "|" for b in blocks if b]
        else:
            return text.strip().split()

    @classmethod
    def train(
        cls,
        texts: List[str],
        vocab_size: int = 8000,
        special_tokens: Optional[List[str]] = None,
        min_freq: int = 2,
        is_cipher: bool = False,
        max_train_samples: int = 1000,
    ) -> BPETokenizer:
        """Train BPE merge rules from scratch with batched multi-merge optimization."""
        if special_tokens is None:
            special_tokens = DEFAULT_SPECIAL_TOKENS

        sample_texts = texts[:max_train_samples] if len(texts) > max_train_samples else texts

        # 1. Pre-tokenize and count frequencies
        word_counts: Dict[str, int] = collections.Counter()
        for text in sample_texts:
            tokens = cls._pre_tokenize(text, is_cipher)
            for tok in tokens:
                word_counts[tok] += 1

        # 2. Build initial vocabulary and symbol splits
        vocab: Dict[str, int] = {tok: idx for idx, tok in enumerate(special_tokens)}
        splits: Dict[str, List[str]] = {}

        for w in word_counts:
            if is_cipher:
                symbols = [w]
            else:
                symbols = list(w) + ["</w>"]
            splits[w] = symbols
            for char in symbols:
                if char not in vocab:
                    vocab[char] = len(vocab)

        merges: List[Tuple[str, str]] = []
        batch_merge_size = 100

        # For plaintext or multi-token sequences, perform merges
        if not is_cipher:
            while len(vocab) < vocab_size:
                pairs: Dict[Tuple[str, str], int] = collections.Counter()
                for w, freq in word_counts.items():
                    syms = splits[w]
                    for i in range(len(syms) - 1):
                        pairs[(syms[i], syms[i + 1])] += freq

                if not pairs:
                    break

                sorted_pairs = pairs.most_common(batch_merge_size * 2)
                if not sorted_pairs or sorted_pairs[0][1] < min_freq:
                    break

                picked_pairs: Dict[Tuple[str, str], str] = {}
                used_symbols: Set[str] = set()

                for pair, count in sorted_pairs:
                    if count < min_freq:
                        break
                    if len(vocab) + len(picked_pairs) >= vocab_size:
                        break
                    if pair[0] not in used_symbols and pair[1] not in used_symbols:
                        merged_symbol = "".join(pair)
                        picked_pairs[pair] = merged_symbol
                        used_symbols.add(pair[0])
                        used_symbols.add(pair[1])
                        if len(picked_pairs) >= batch_merge_size:
                            break

                # If no frequent non-overlapping pairs found, stop
                if not picked_pairs:
                    break

                for pair, merged_symbol in picked_pairs.items():
                    merges.append(pair)
                    vocab[merged_symbol] = len(vocab)

                new_splits: Dict[str, List[str]] = {}
                for w, syms in splits.items():
                    new_syms = []
                    i = 0
                    while i < len(syms):
                        if i < len(syms) - 1 and (syms[i], syms[i + 1]) in picked_pairs:
                            new_syms.append(picked_pairs[(syms[i], syms[i + 1])])
                            i += 2
                        else:
                            new_syms.append(syms[i])
                            i += 1
                    new_splits[w] = new_syms
                splits = new_splits
        else:
            # Cipher BPE: merge adjacent 8-bit blocks into multi-byte subwords
            seq_counts: Dict[Tuple[str, ...], int] = collections.Counter()
            for text in sample_texts:
                blocks = tuple(cls._pre_tokenize(text, is_cipher=True))
                if blocks:
                    seq_counts[blocks] += 1

            cipher_splits = {seq: list(seq) for seq in seq_counts}

            while len(vocab) < vocab_size:
                pairs: Dict[Tuple[str, str], int] = collections.Counter()
                for seq, freq in seq_counts.items():
                    syms = cipher_splits[seq]
                    for i in range(len(syms) - 1):
                        pairs[(syms[i], syms[i + 1])] += freq

                if not pairs:
                    break

                sorted_pairs = pairs.most_common(batch_merge_size * 2)
                if not sorted_pairs or sorted_pairs[0][1] < min_freq:
                    break

                picked_pairs: Dict[Tuple[str, str], str] = {}
                used_symbols: Set[str] = set()

                for pair, count in sorted_pairs:
                    if count < min_freq:
                        break
                    if len(vocab) + len(picked_pairs) >= vocab_size:
                        break
                    if pair[0] not in used_symbols and pair[1] not in used_symbols:
                        merged_symbol = "".join(pair)
                        picked_pairs[pair] = merged_symbol
                        used_symbols.add(pair[0])
                        used_symbols.add(pair[1])
                        if len(picked_pairs) >= batch_merge_size:
                            break

                # If no frequent non-overlapping pairs found, stop
                if not picked_pairs:
                    break

                for pair, merged_symbol in picked_pairs.items():
                    merges.append(pair)
                    vocab[merged_symbol] = len(vocab)

                new_cipher_splits: Dict[Tuple[str, ...], List[str]] = {}
                for seq, syms in cipher_splits.items():
                    new_syms = []
                    i = 0
                    while i < len(syms):
                        if i < len(syms) - 1 and (syms[i], syms[i + 1]) in picked_pairs:
                            new_syms.append(picked_pairs[(syms[i], syms[i + 1])])
                            i += 2
                        else:
                            new_syms.append(syms[i])
                            i += 1
                    new_cipher_splits[seq] = new_syms
                cipher_splits = new_cipher_splits

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens, is_cipher=is_cipher)

    def _tokenize_word(self, word: str) -> List[str]:
        if word in self.cache:
            return self.cache[word]

        if self.is_cipher:
            split = [word]
        else:
            split = list(word) + ["</w>"]

        while len(split) > 1:
            pairs = [(split[i], split[i + 1]) for i in range(len(split) - 1)]
            candidate_pairs = [p for p in pairs if p in self.bpe_ranks]
            if not candidate_pairs:
                break

            best_pair = min(candidate_pairs, key=lambda p: self.bpe_ranks[p])
            merged_symbol = "".join(best_pair)

            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and (split[i], split[i + 1]) == best_pair:
                    new_split.append(merged_symbol)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            split = new_split

        self.cache[word] = split
        return split

    def encode(self, text: str, add_special_tokens: bool = True) -> EncodedOutput:
        """Encodes text to subword token IDs (with BOS and EOS added)."""
        tokens = []
        if self.is_cipher:
            blocks = self._pre_tokenize(text, is_cipher=True)
            syms = list(blocks)
            while len(syms) > 1:
                pairs = [(syms[i], syms[i + 1]) for i in range(len(syms) - 1)]
                candidates = [p for p in pairs if p in self.bpe_ranks]
                if not candidates:
                    break
                best_pair = min(candidates, key=lambda p: self.bpe_ranks[p])
                merged = "".join(best_pair)
                new_syms = []
                i = 0
                while i < len(syms):
                    if i < len(syms) - 1 and (syms[i], syms[i + 1]) == best_pair:
                        new_syms.append(merged)
                        i += 2
                    else:
                        new_syms.append(syms[i])
                        i += 1
                syms = new_syms
            tokens = syms
        else:
            words = self._pre_tokenize(text, is_cipher=False)
            for w in words:
                tokens.extend(self._tokenize_word(w))

        unk_id = self.vocab.get(self.unk_token, 3)
        ids = [self.vocab.get(tok, unk_id) for tok in tokens]

        if add_special_tokens:
            bos_id = self.vocab.get(self.bos_token, 1)
            eos_id = self.vocab.get(self.eos_token, 2)
            ids = [bos_id] + ids + [eos_id]

        return EncodedOutput(ids, tokens)

    def decode(self, ids: List[int]) -> str:
        """Decodes token IDs back to a reconstructed string."""
        special_ids = {
            self.vocab.get(st) for st in self.special_tokens if st in self.vocab
        }
        tokens = []
        for token_id in ids:
            if token_id in special_ids:
                continue
            tok = self.inv_vocab.get(token_id, "")
            tokens.append(tok)

        if self.is_cipher:
            return "".join(tokens)
        else:
            text = "".join(tokens).replace("</w>", " ").strip()
            return text

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def token_to_id(self, token: str) -> Optional[int]:
        return self.vocab.get(token, None)

    def id_to_token(self, idx: int) -> Optional[str]:
        return self.inv_vocab.get(idx, None)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "is_cipher": self.is_cipher,
            "special_tokens": self.special_tokens,
            "vocab": self.vocab,
            "merges": [list(pair) for pair in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_file(cls, path: str) -> BPETokenizer:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        is_cipher = data.get("is_cipher", False)

        # 1. Custom scratch format
        if "vocab" in data:
            vocab = data["vocab"]
            merges = [tuple(pair) for pair in data.get("merges", [])]
            special_tokens = data.get("special_tokens", DEFAULT_SPECIAL_TOKENS)
            return cls(vocab=vocab, merges=merges, special_tokens=special_tokens, is_cipher=is_cipher)

        # 2. HuggingFace format compatibility
        if "model" in data and "vocab" in data["model"]:
            vocab = data["model"]["vocab"]
            raw_merges = data["model"].get("merges", [])
            merges = []
            for m in raw_merges:
                if isinstance(m, str):
                    parts = m.split()
                    if len(parts) == 2:
                        merges.append((parts[0], parts[1]))
                elif isinstance(m, (list, tuple)) and len(m) == 2:
                    merges.append((m[0], m[1]))
            special_tokens = data.get("special_tokens", DEFAULT_SPECIAL_TOKENS)
            return cls(vocab=vocab, merges=merges, special_tokens=special_tokens, is_cipher=is_cipher)

        raise ValueError(f"Unrecognized tokenizer file format at: {path}")


def train_tokenizer(
    input_file: str = "data/brown_plain.txt",
    output_file: str = "data/brown_tokenizer.json",
    vocab_size: int = 8000,
) -> BPETokenizer:
    with open(input_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    tok = BPETokenizer.train(texts, vocab_size=vocab_size)
    tok.save(output_file)
    return tok


def load_tokenizer(
    tokenizer_file: str = "data/brown_tokenizer.json",
) -> BPETokenizer:
    return BPETokenizer.from_file(tokenizer_file)