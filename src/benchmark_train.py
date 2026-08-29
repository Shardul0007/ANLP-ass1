import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import BinaryToTextDataset, collate_fn
from .models.masks import create_causal_mask
from .models.transformer import BinaryToTextTransformer
from .bucketing import LengthBucketBatchSampler

# =========================================
# Configuration
# =========================================

NUM_BATCHES = 10

MAX_CIPHER_LENGTH = 8192

VOCAB_SIZE = 8000

D_MODEL = 256
NUM_HEADS = 8
D_FF = 1024
NUM_LAYERS = 2

LEARNING_RATE = 3e-4


# =========================================
# Dataset
# =========================================

dataset = BinaryToTextDataset(
    "data/brown_cipher.txt",
    "data/brown_plain.txt",
)

batch_sampler = LengthBucketBatchSampler(
    dataset=dataset,
    batch_sizes=[
        4,  # <= 2048
        2,  # <= 4096
        1,  # <= 8192
        1,  # > 8192
    ],
    boundaries=[
        2048,
        4096,
        8192,
    ],
    shuffle=True,
    seed=42,
)

loader = DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    collate_fn=collate_fn,
)


# =========================================
# Model
# =========================================

model = BinaryToTextTransformer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
    num_layers=NUM_LAYERS,
    max_cipher_length=MAX_CIPHER_LENGTH,
    max_text_length=1024,
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)

criterion = nn.CrossEntropyLoss(
    ignore_index=0,
)


# =========================================
# Benchmark
# =========================================

model.train()

start_time = time.perf_counter()

total_loss = 0.0
total_examples = 0
for batch_index, batch in enumerate(loader):

    if batch_index >= NUM_BATCHES:
        break
    cipher = batch["cipher"][
        :, :MAX_CIPHER_LENGTH
    ]

    total_examples += cipher.size(0)
    cipher_padding_mask = (
        batch["cipher_padding_mask"][
            :, :MAX_CIPHER_LENGTH
        ]
        .unsqueeze(1)
        .unsqueeze(2)
    )

    decoder_input = batch[
        "decoder_input"
    ]

    target = batch["target"]

    decoder_padding_mask = (
        batch["target_padding_mask"]
        .unsqueeze(1)
        .unsqueeze(2)
    )

    causal_mask = create_causal_mask(
        decoder_input.size(1),
        decoder_input.device,
    )

    # Forward
    output = model(
        cipher,
        decoder_input,
        cipher_padding_mask=cipher_padding_mask,
        decoder_self_attention_mask=causal_mask,
        decoder_padding_mask=decoder_padding_mask,
    )

    loss = criterion(
        output["logits"].reshape(
            -1,
            VOCAB_SIZE,
        ),
        target.reshape(-1),
    )

    # Backward
    optimizer.zero_grad()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )

    optimizer.step()

    total_loss += loss.item()

    print(
        f"Batch {batch_index + 1:2d} | "
        f"Batch size: {cipher.size(0)} | "
        f"Max cipher: {cipher.size(1):5d} | "
        f"Loss: {loss.item():.4f}"
    )


elapsed = time.perf_counter() - start_time

completed = min(
    NUM_BATCHES,
    len(loader),
)

print("\n" + "=" * 60)

print(
    f"Completed batches: {completed}"
)

print(
    f"Total time: {elapsed:.2f} seconds"
)

print(
    f"Average time/batch: "
    f"{elapsed / completed:.2f} seconds"
)
print(
    f"Total examples: {total_examples}"
)

print(
    f"Time/example: "
    f"{elapsed / total_examples:.2f} seconds"
)

print(
    f"Average loss: "
    f"{total_loss / completed:.4f}"
)

print("=" * 60)