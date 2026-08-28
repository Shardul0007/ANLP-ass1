from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[BOS]",
    "[EOS]",
]


def train_tokenizer(
    input_file="data/brown_plain.txt",
    output_file="data/brown_tokenizer.json",
    vocab_size=8000,
):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
    )

    tokenizer.train([input_file], trainer)

    tokenizer.save(output_file)

    print(f"Tokenizer saved to: {output_file}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    return tokenizer


def load_tokenizer(
    tokenizer_file="data/brown_tokenizer.json",
):
    return Tokenizer.from_file(tokenizer_file)


if __name__ == "__main__":
    tokenizer = train_tokenizer()

    test_text = (
        "Robert Boulter is an English film television "
        "and theatre actor"
    )

    encoded = tokenizer.encode(test_text)

    print("\nOriginal:")
    print(test_text)

    print("\nTokens:")
    print(encoded.tokens)

    print("\nIDs:")
    print(encoded.ids)

    print("\nDecoded:")
    print(tokenizer.decode(encoded.ids))