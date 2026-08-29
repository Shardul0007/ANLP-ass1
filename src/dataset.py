from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from .tokenizer import load_tokenizer


class BinaryToTextDataset(Dataset):
    def __init__(
        self,
        cipher_file,
        plain_file,
        tokenizer_file="data/brown_tokenizer.json",
        max_cipher_len=None,
    ):
        self.cipher_file = Path(cipher_file)
        self.plain_file = Path(plain_file)
        self.max_cipher_len = max_cipher_len

        with open(self.cipher_file, "r", encoding="utf-8") as f:
            self.cipher_lines = [line.strip() for line in f]

        with open(self.plain_file, "r", encoding="utf-8") as f:
            self.plain_lines = [line.strip() for line in f]

        if len(self.cipher_lines) != len(self.plain_lines):
            raise ValueError(
                f"Dataset size mismatch: "
                f"{len(self.cipher_lines)} cipher lines vs "
                f"{len(self.plain_lines)} plaintext lines"
            )

        if len(self.cipher_lines) == 0:
            raise ValueError("Dataset is empty")

        for i, sequence in enumerate(self.cipher_lines):
            if not sequence:
                raise ValueError(
                    f"Empty cipher sequence at line {i}"
                )

            if any(bit not in "01" for bit in sequence):
                raise ValueError(
                    f"Invalid character in cipher sequence at line {i}"
                )

        self.tokenizer = load_tokenizer(tokenizer_file)

        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.unk_id = self.tokenizer.token_to_id("[UNK]")
        self.bos_id = self.tokenizer.token_to_id("[BOS]")
        self.eos_id = self.tokenizer.token_to_id("[EOS]")

    def __len__(self):
        return len(self.cipher_lines)

    def __getitem__(self, index):
        cipher = self.cipher_lines[index]
        plaintext = self.plain_lines[index]

        # Aligned truncation if max_cipher_len is set
        if self.max_cipher_len is not None and len(cipher) > self.max_cipher_len:
            max_bytes = self.max_cipher_len // 8
            cipher = cipher[: max_bytes * 8]
            plaintext = plaintext[:max_bytes]

        # Binary input (list of 0s and 1s).
        cipher_ids = [int(bit) for bit in cipher]

        # Tokenize plaintext.
        encoded = self.tokenizer.encode(plaintext)
        token_ids = encoded.ids

        # Add BOS and EOS.
        decoder_input = [self.bos_id] + token_ids
        target = token_ids + [self.eos_id]

        return {
            "cipher": torch.tensor(
                cipher_ids,
                dtype=torch.long,
            ),
            "decoder_input": torch.tensor(
                decoder_input,
                dtype=torch.long,
            ),
            "target": torch.tensor(
                target,
                dtype=torch.long,
            ),
            "plain_text": plaintext,
        }


def collate_fn(batch):
    cipher_sequences = [
        item["cipher"] for item in batch
    ]

    decoder_inputs = [
        item["decoder_input"] for item in batch
    ]

    targets = [
        item["target"] for item in batch
    ]

    # -------------------------
    # Cipher padding
    # -------------------------

    max_cipher_length = max(
        len(sequence)
        for sequence in cipher_sequences
    )

    padded_cipher = torch.full(
        (len(batch), max_cipher_length),
        fill_value=2,
        dtype=torch.long,
    )

    cipher_padding_mask = torch.ones(
        (len(batch), max_cipher_length),
        dtype=torch.bool,
    )

    for i, sequence in enumerate(cipher_sequences):
        length = len(sequence)

        padded_cipher[i, :length] = sequence
        cipher_padding_mask[i, :length] = False

    # -------------------------
    # Decoder padding
    # -------------------------

    max_target_length = max(
        len(sequence)
        for sequence in decoder_inputs
    )

    padded_decoder_input = torch.full(
        (len(batch), max_target_length),
        fill_value=0,
        dtype=torch.long,
    )

    padded_target = torch.full(
        (len(batch), max_target_length),
        fill_value=0,
        dtype=torch.long,
    )

    target_padding_mask = torch.ones(
        (len(batch), max_target_length),
        dtype=torch.bool,
    )

    for i in range(len(batch)):
        length = len(decoder_inputs[i])

        padded_decoder_input[i, :length] = decoder_inputs[i]
        padded_target[i, :length] = targets[i]

        target_padding_mask[i, :length] = False

    result = {
        "cipher": padded_cipher,
        "cipher_padding_mask": cipher_padding_mask,
        "decoder_input": padded_decoder_input,
        "target": padded_target,
        "target_padding_mask": target_padding_mask,
    }
    if "plain_text" in batch[0]:
        result["plain_text"] = [item["plain_text"] for item in batch]
    return result


if __name__ == "__main__":
    dataset = BinaryToTextDataset(
        "data/brown_cipher.txt",
        "data/brown_plain.txt",
    )

    print("Dataset size:", len(dataset))

    sample = dataset[0]

    print("\nFirst example:")
    print("Cipher length:", len(sample["cipher"]))

    print("Decoder input:")
    print(sample["decoder_input"][:20])

    print("Target:")
    print(sample["target"][:20])

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
    )

    batch = next(iter(loader))

    print("\nBatch:")
    print("Cipher shape:", batch["cipher"].shape)
    print(
        "Decoder input shape:",
        batch["decoder_input"].shape,
    )
    print("Target shape:", batch["target"].shape)