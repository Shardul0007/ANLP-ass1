import os
import json
from pathlib import Path
from typing import Optional, Tuple
import random

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Auto-detect data directory: prefer data/ if present, fallback to Dataset_A1
if os.path.exists(os.path.join(_PROJECT_ROOT, "data", "brown_cipher.txt")):
    DATASET_DIR = os.path.join(_PROJECT_ROOT, "data")
else:
    DATASET_DIR = os.path.join(_PROJECT_ROOT, "Dataset_A1")

CACHE_DIR = os.path.join(_PROJECT_ROOT, ".cache")

PLAIN_CHUNK_SIZE = 128
MIN_CHUNK_CHARS = 32

# BPE special tokens
BPE_PAD = "<pad>"
BPE_BOS = "<bos>"
BPE_EOS = "<eos>"
BPE_UNK = "<unk>"

# Byte-level (BLT) special tokens
BYTE_PAD = 256
BYTE_BOS = 257
BYTE_EOS = 258
BYTE_VOCAB_SIZE = 259


def chunk_dataset(
    cipher_lines: list[str], plain_lines: list[str], plain_chunk_size: int = PLAIN_CHUNK_SIZE
) -> tuple[list[str], list[str]]:
    """Chunk dataset aligned by character.
    Each plaintext character corresponds to 8 cipher bits.
    Inserts a '|' separator after every 8 bits in the cipher chunk.
    """
    assert len(cipher_lines) == len(plain_lines), (
        f"Cipher ({len(cipher_lines)}) and plain ({len(plain_lines)}) line counts don't match"
    )
    chunked_cipher = []
    chunked_plain = []

    for c, p in zip(cipher_lines, plain_lines):
        for i in range(0, len(p), plain_chunk_size):
            p_chunk = p[i : i + plain_chunk_size]
            c_chunk_raw = c[i * 8 : (i + plain_chunk_size) * 8]

            if len(p_chunk) < MIN_CHUNK_CHARS:
                continue

            c_chunk = ""
            for j in range(0, len(c_chunk_raw), 8):
                c_chunk += c_chunk_raw[j : j + 8] + "|"

            if len(p_chunk) > 0:
                chunked_plain.append(p_chunk)
                chunked_cipher.append(c_chunk)

    return chunked_cipher, chunked_plain


def load_raw_lines(data_dir: str) -> tuple[list[str], list[str]]:
    cipher_path = os.path.join(data_dir, "brown_cipher.txt")
    plain_path = os.path.join(data_dir, "brown_plain.txt")

    with open(cipher_path, "r", encoding="utf-8") as f:
        cipher_lines = [line.strip() for line in f if line.strip()]

    with open(plain_path, "r", encoding="utf-8") as f:
        plain_lines = [line.strip() for line in f if line.strip()]

    return cipher_lines, plain_lines


def split_data(
    cipher_lines: list[str],
    plain_lines: list[str],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict:
    """Split data into train/val/test with fixed seed (80/10/10)."""
    n = len(cipher_lines)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    splits = {}
    for name, idx_list in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        c_lines = [cipher_lines[i] for i in idx_list]
        p_lines = [plain_lines[i] for i in idx_list]
        c_chunked, p_chunked = chunk_dataset(c_lines, p_lines, plain_chunk_size=PLAIN_CHUNK_SIZE)
        splits[name] = {"cipher": c_chunked, "plain": p_chunked}

    return splits


def get_split_data_cached(data_dir: str, cache_dir: str, seed: int = 42) -> dict:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"splits_v3_{seed}.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    cipher_lines, plain_lines = load_raw_lines(data_dir)
    splits = split_data(cipher_lines, plain_lines, seed=seed)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(splits, f)

    return splits


def train_single_tokenizer(texts: list[str], vocab_size: int, save_path: str) -> Tokenizer:
    if os.path.exists(save_path):
        return Tokenizer.from_file(save_path)

    tokenizer = Tokenizer(BPE(unk_token=BPE_UNK))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=[BPE_PAD, BPE_BOS, BPE_EOS, BPE_UNK], show_progress=True)
    tokenizer.train_from_iterator(texts, trainer=trainer)

    bos_id = tokenizer.token_to_id(BPE_BOS)
    eos_id = tokenizer.token_to_id(BPE_EOS)
    tokenizer.post_processor = TemplateProcessing(
        single=f"{BPE_BOS}:0 $A:0 {BPE_EOS}:0",
        special_tokens=[(BPE_BOS, bos_id), (BPE_EOS, eos_id)],
    )
    pad_id = tokenizer.token_to_id(BPE_PAD)
    tokenizer.enable_padding(pad_id=pad_id, pad_token=BPE_PAD)
    tokenizer.save(save_path)

    return tokenizer


def train_bpe_tokenizers(
    train_cipher: list[str],
    train_plain: list[str],
    vocab_size: int = 8000,
    cache_dir: str = CACHE_DIR,
) -> tuple[Tokenizer, Tokenizer]:
    """Train separate BPE tokenizers for source cipher and target plaintext."""
    os.makedirs(cache_dir, exist_ok=True)
    src_path = os.path.join(cache_dir, "bpe_tokenizer_src_v3.json")
    tgt_path = os.path.join(cache_dir, "bpe_tokenizer_tgt_v3.json")
    src_tokenizer = train_single_tokenizer(train_cipher, vocab_size, src_path)
    tgt_tokenizer = train_single_tokenizer(train_plain, vocab_size, tgt_path)

    return src_tokenizer, tgt_tokenizer


def get_bpe_special_ids(tokenizer: Tokenizer) -> dict:
    return {
        "pad": tokenizer.token_to_id(BPE_PAD),
        "bos": tokenizer.token_to_id(BPE_BOS),
        "eos": tokenizer.token_to_id(BPE_EOS),
    }


class CipherDatasetTokenized(Dataset):
    """Dataset for tokenized mode (C1–C4).
    Source: BPE-tokenized cipher binary strings (with '|' separators).
    Target: BPE-tokenized plaintext.
    """

    def __init__(
        self,
        cipher_lines: list[str],
        plain_lines: list[str],
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        max_seq_len: int = 512,
    ):
        self.cipher_lines = cipher_lines
        self.plain_lines = plain_lines
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.cipher_lines)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cipher = self.cipher_lines[idx]
        plain = self.plain_lines[idx]

        src_enc = self.src_tokenizer.encode(cipher)
        tgt_enc = self.tgt_tokenizer.encode(plain)

        src_ids = src_enc.ids[: self.max_seq_len]
        tgt_ids = tgt_enc.ids[: self.max_seq_len]

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


class CipherDatasetTokenFree(Dataset):
    """Dataset for token-free mode (C5 — BLT).
    Source: raw byte tensors of formatted cipher binary strings.
    Target: raw byte tensors of plaintext.
    """

    def __init__(
        self,
        cipher_lines: list[str],
        plain_lines: list[str],
        max_byte_len: int = 2048,
    ):
        self.cipher_lines = cipher_lines
        self.plain_lines = plain_lines
        self.max_byte_len = max_byte_len

    def __len__(self) -> int:
        return len(self.cipher_lines)

    def _str_to_bytes(self, s: str, max_len: int) -> torch.Tensor:
        byte_vals = list(s.encode("utf-8"))[: max_len - 2]
        return torch.tensor([BYTE_BOS] + byte_vals + [BYTE_EOS], dtype=torch.long)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cipher = self.cipher_lines[idx]
        plain = self.plain_lines[idx]

        src = self._str_to_bytes(cipher, self.max_byte_len)
        tgt = self._str_to_bytes(plain, self.max_byte_len)
        return src, tgt


def collate_tokenized(batch: list[tuple[torch.Tensor, torch.Tensor]], pad_id: int = 0):
    """Pad sequences in a batch to the same length."""
    src_list, tgt_list = zip(*batch)
    src_max = max(s.size(0) for s in src_list)
    tgt_max = max(t.size(0) for t in tgt_list)

    src_batch = torch.full((len(batch), src_max), pad_id, dtype=torch.long)
    tgt_batch = torch.full((len(batch), tgt_max), pad_id, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_list, tgt_list)):
        src_batch[i, : s.size(0)] = s
        tgt_batch[i, : t.size(0)] = t

    return src_batch, tgt_batch


def collate_token_free(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    return collate_tokenized(batch, pad_id=BYTE_PAD)


def build_dataloaders(
    tokenization: str = "subword",
    batch_size: int = 64,
    max_seq_len: int = 512,
    vocab_size: int = 8000,
    seed: int = 42,
    num_workers: int = 0,
    data_dir: str = DATASET_DIR,
) -> dict:
    """Build train/val/test dataloaders and tokenizer info."""
    splits = get_split_data_cached(data_dir, CACHE_DIR, seed=seed)

    if tokenization == "subword":
        src_tokenizer, tgt_tokenizer = train_bpe_tokenizers(
            splits["train"]["cipher"], splits["train"]["plain"], vocab_size=vocab_size
        )
        special_ids = get_bpe_special_ids(src_tokenizer)
        datasets = {}

        for split_name in ["train", "val", "test"]:
            datasets[split_name] = CipherDatasetTokenized(
                splits[split_name]["cipher"],
                splits[split_name]["plain"],
                src_tokenizer,
                tgt_tokenizer,
                max_seq_len,
            )
        collate_fn = lambda batch: collate_tokenized(batch, pad_id=special_ids["pad"])
        info = {
            "src_vocab_size": src_tokenizer.get_vocab_size(),
            "tgt_vocab_size": tgt_tokenizer.get_vocab_size(),
            "pad_idx": special_ids["pad"],
            "bos_idx": special_ids["bos"],
            "eos_idx": special_ids["eos"],
            "tokenizer_src": src_tokenizer,
            "tokenizer_tgt": tgt_tokenizer,
        }
    elif tokenization == "blt":
        max_byte_len = max_seq_len
        datasets = {}

        for split_name in ["train", "val", "test"]:
            datasets[split_name] = CipherDatasetTokenFree(
                splits[split_name]["cipher"], splits[split_name]["plain"], max_byte_len
            )
        collate_fn = collate_token_free
        info = {
            "src_vocab_size": BYTE_VOCAB_SIZE,
            "tgt_vocab_size": BYTE_VOCAB_SIZE,
            "pad_idx": BYTE_PAD,
            "bos_idx": BYTE_BOS,
            "eos_idx": BYTE_EOS,
            "tokenizer_src": None,
            "tokenizer_tgt": None,
        }
    else:
        raise ValueError(f"Unknown tokenization mode: {tokenization}")

    train_loader = DataLoader(
        datasets["train"], batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        datasets["val"], batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        datasets["test"], batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "splits": splits,
        **info,
    }