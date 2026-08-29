import time

import torch

from .models.transformer import BinaryToTextTransformer


def benchmark(sequence_length):
    print(
        f"\nTesting sequence length: {sequence_length}"
    )

    model = BinaryToTextTransformer(
        vocab_size=8000,
        d_model=256,
        num_heads=8,
        d_ff=1024,
        num_layers=2,
        max_cipher_length=sequence_length,
        max_text_length=1024,
        dropout=0.0,
    )

    model.eval()

    cipher = torch.randint(
        0,
        2,
        (1, sequence_length),
    )

    decoder_input = torch.randint(
        0,
        8000,
        (1, 8),
    )

    start = time.perf_counter()

    with torch.no_grad():
        output = model(
            cipher,
            decoder_input,
        )

    elapsed = time.perf_counter() - start

    print(
        f"Time: {elapsed:.3f} seconds"
    )

    print(
        "Logits:",
        output["logits"].shape,
    )


if __name__ == "__main__":
    lengths = [
    2048,
    4096,
    8192,
]

    for length in lengths:
        benchmark(length)