import torch
from torch.utils.data import random_split


def split_dataset(
    dataset,
    train_ratio=0.9,
    seed=42,
):
    train_size = int(
        len(dataset) * train_ratio
    )

    validation_size = (
        len(dataset) - train_size
    )

    generator = torch.Generator().manual_seed(
        seed
    )

    train_dataset, validation_dataset = (
        random_split(
            dataset,
            [train_size, validation_size],
            generator=generator,
        )
    )

    return train_dataset, validation_dataset


def save_checkpoint(
    model,
    optimizer,
    epoch,
    validation_loss,
    path,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_loss": validation_loss,
    }

    torch.save(
        checkpoint,
        path,
    )

    print(
        f"Checkpoint saved to: {path}"
    )