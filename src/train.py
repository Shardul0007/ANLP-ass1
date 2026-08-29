import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

from .dataset import (
    BinaryToTextDataset,
    collate_fn,
)
from .models.masks import create_causal_mask
from .models.transformer import (
    BinaryToTextTransformer,
)
from .training import (
    save_checkpoint,
    split_dataset,
)


# =========================================
# Configuration
# =========================================

BATCH_SIZE = 1

EPOCHS = 1

MAX_CIPHER_LENGTH = 4096

VOCAB_SIZE = 8000

D_MODEL = 256
NUM_HEADS = 8
D_FF = 1024
NUM_LAYERS = 2

LEARNING_RATE = 3e-4

CHECKPOINT_DIR = "checkpoints"
MAX_TRAIN_BATCHES = 100
MAX_VALIDATION_BATCHES = 20


# =========================================
# Dataset
# =========================================

dataset = BinaryToTextDataset(
    "data/brown_cipher.txt",
    "data/brown_plain.txt",
)

train_dataset, validation_dataset = (
    split_dataset(
        dataset,
        train_ratio=0.9,
        seed=42,
    )
)

print(
    "Training examples:",
    len(train_dataset),
)

print(
    "Validation examples:",
    len(validation_dataset),
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
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
    max_text_length=512,
).to(device)


# =========================================
# Optimizer
# =========================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)

criterion = nn.CrossEntropyLoss(
    ignore_index=0,
)


# =========================================
# Training function
# =========================================

def train_epoch():
    model.train()

    total_loss = 0.0

    for batch_index, batch in enumerate(
        train_loader
    ):
        if batch_index >= MAX_TRAIN_BATCHES:
            break
        cipher = batch["cipher"][
            :, :MAX_CIPHER_LENGTH
        ].to(device)

        cipher_padding_mask = (
            batch["cipher_padding_mask"][
                :, :MAX_CIPHER_LENGTH
            ]
        )

        cipher_padding_mask = (
            cipher_padding_mask
            .unsqueeze(1)
            .unsqueeze(2)
            .to(device)
        )

        decoder_input = (
            batch["decoder_input"]
            .to(device)
        )

        target = (
            batch["target"]
            .to(device)
        )

        decoder_padding_mask = (
            batch["target_padding_mask"]
            .unsqueeze(1)
            .unsqueeze(2)
            .to(device)
        )

        causal_mask = create_causal_mask(
            decoder_input.size(1),
            device,
        )

        output = model(
            cipher,
            decoder_input,
            cipher_padding_mask=(
                cipher_padding_mask
            ),
            decoder_self_attention_mask=(
                causal_mask
            ),
            decoder_padding_mask=(
                decoder_padding_mask
            ),
        )

        logits = output["logits"]

        loss = criterion(
            logits.reshape(
                -1,
                VOCAB_SIZE,
            ),
            target.reshape(-1),
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += loss.item()

        if (batch_index + 1) % 100 == 0:
            print(
                f"  Batch {batch_index + 1} | "
                f"Loss: {loss.item():.4f}"
            )

    return (
        total_loss
        / len(train_loader)
    )


# =========================================
# Validation
# =========================================

@torch.no_grad()
def validate():
    model.eval()

    total_loss = 0.0

    for batch_index, batch in enumerate(validation_loader):
        if batch_index >= MAX_VALIDATION_BATCHES:
            break

        cipher = batch["cipher"][
            :, :MAX_CIPHER_LENGTH
        ].to(device)

        cipher_padding_mask = (
            batch["cipher_padding_mask"][
                :, :MAX_CIPHER_LENGTH
            ]
        )

        cipher_padding_mask = (
            cipher_padding_mask
            .unsqueeze(1)
            .unsqueeze(2)
            .to(device)
        )

        decoder_input = (
            batch["decoder_input"]
            .to(device)
        )

        target = (
            batch["target"]
            .to(device)
        )

        decoder_padding_mask = (
            batch["target_padding_mask"]
            .unsqueeze(1)
            .unsqueeze(2)
            .to(device)
        )

        causal_mask = create_causal_mask(
            decoder_input.size(1),
            device,
        )

        output = model(
            cipher,
            decoder_input,
            cipher_padding_mask=(
                cipher_padding_mask
            ),
            decoder_self_attention_mask=(
                causal_mask
            ),
            decoder_padding_mask=(
                decoder_padding_mask
            ),
        )

        loss = criterion(
            output["logits"].reshape(
                -1,
                VOCAB_SIZE,
            ),
            target.reshape(-1),
        )

        total_loss += loss.item()

    return (
        total_loss
        / len(validation_loader)
    )


# =========================================
# Training loop
# =========================================

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True,
)

for epoch in range(
    1,
    EPOCHS + 1,
):

    print(
        f"\n========== Epoch {epoch} =========="
    )

    train_loss = train_epoch()

    validation_loss = validate()

    print(
        f"\nEpoch {epoch}"
    )

    print(
        f"Train loss: {train_loss:.4f}"
    )

    print(
        f"Validation loss: "
        f"{validation_loss:.4f}"
    )

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"c1_epoch_{epoch}.pt",
    )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        validation_loss=validation_loss,
        path=checkpoint_path,
    )