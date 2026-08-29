"""
Main training loop for Configuration 1 (C1: Base Transformer) with WandB logging.
"""

import argparse
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .bucketing import LengthBucketBatchSampler
from .dataset import BinaryToTextDataset, collate_fn
from .models.masks import create_causal_mask
from .models.transformer import BinaryToTextTransformer
from .training import save_checkpoint, split_dataset
from .utils import compute_all_metrics, plot_training_curves


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.05
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def parse_args():
    parser = argparse.ArgumentParser(description="Train C1 Base Transformer")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Base batch size")
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch = batch_size * accum_steps)",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument(
        "--warmup_steps", type=int, default=300, help="Linear warmup steps"
    )
    parser.add_argument(
        "--max_cipher_len",
        type=int,
        default=8192,
        help="Max cipher length in bits (aligned truncation for longer)",
    )
    parser.add_argument(
        "--max_text_len", type=int, default=1024, help="Max plaintext token length"
    )
    parser.add_argument(
        "--d_model", type=int, default=256, help="Transformer model dimension"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--d_ff", type=int, default=1024, help="Feed-forward network hidden dimension"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of encoder/decoder layers"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--use_bucketing",
        action="store_true",
        default=True,
        help="Use length-bucketed batch sampling",
    )
    parser.add_argument(
        "--no_bucketing",
        action="store_false",
        dest="use_bucketing",
        help="Disable length bucketing",
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=30,
        help="Number of validation samples for greedy metric evaluation per epoch",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="anlp-assignment1",
        help="WandB project name",
    )
    parser.add_argument(
        "--run_name", type=str, default="C1-Baseline", help="Run name for WandB"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=None,
        help="Optional limit on train batches per epoch (for quick debugging)",
    )
    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=None,
        help="Optional limit on val batches (for quick debugging)",
    )

    return parser.parse_args()


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    device,
    accum_steps=1,
    max_batches=None,
    vocab_size=8000,
    epoch=1,
    wandb_run=None,
):
    model.train()
    total_loss = 0.0
    num_batches = len(loader) if max_batches is None else min(len(loader), max_batches)

    optimizer.zero_grad()
    epoch_start = time.perf_counter()

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        cipher = batch["cipher"].to(device)
        cipher_padding_mask = (
            batch["cipher_padding_mask"].unsqueeze(1).unsqueeze(2).to(device)
        )
        decoder_input = batch["decoder_input"].to(device)
        target = batch["target"].to(device)
        decoder_padding_mask = (
            batch["target_padding_mask"].unsqueeze(1).unsqueeze(2).to(device)
        )
        causal_mask = create_causal_mask(decoder_input.size(1), device)

        output = model(
            cipher,
            decoder_input,
            cipher_padding_mask=cipher_padding_mask,
            decoder_self_attention_mask=causal_mask,
            decoder_padding_mask=decoder_padding_mask,
        )

        logits = output["logits"]
        loss = criterion(
            logits.reshape(-1, vocab_size),
            target.reshape(-1),
        )

        # Scale loss for gradient accumulation
        scaled_loss = loss / accum_steps
        scaled_loss.backward()

        if (batch_index + 1) % accum_steps == 0 or (batch_index + 1) == num_batches:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()

        if (batch_index + 1) % 100 == 0 or (batch_index + 1) == num_batches:
            current_lr = scheduler.get_last_lr()[0]
            avg_loss_so_far = total_loss / (batch_index + 1)
            elapsed = time.perf_counter() - epoch_start
            print(
                f"Epoch {epoch} | Batch {batch_index + 1:4d}/{num_batches} | "
                f"Loss: {loss.item():.4f} (Avg: {avg_loss_so_far:.4f}) | "
                f"LR: {current_lr:.2e} | Elapsed: {elapsed:.1f}s"
            )

            if wandb_run is not None:
                try:
                    wandb_run.log(
                        {
                            "train/batch_loss": loss.item(),
                            "train/learning_rate": current_lr,
                        }
                    )
                except Exception:
                    pass

    return total_loss / num_batches


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device, vocab_size=8000, max_batches=None):
    model.eval()
    total_loss = 0.0
    num_batches = len(loader) if max_batches is None else min(len(loader), max_batches)

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        cipher = batch["cipher"].to(device)
        cipher_padding_mask = (
            batch["cipher_padding_mask"].unsqueeze(1).unsqueeze(2).to(device)
        )
        decoder_input = batch["decoder_input"].to(device)
        target = batch["target"].to(device)
        decoder_padding_mask = (
            batch["target_padding_mask"].unsqueeze(1).unsqueeze(2).to(device)
        )
        causal_mask = create_causal_mask(decoder_input.size(1), device)

        output = model(
            cipher,
            decoder_input,
            cipher_padding_mask=cipher_padding_mask,
            decoder_self_attention_mask=causal_mask,
            decoder_padding_mask=decoder_padding_mask,
        )

        loss = criterion(
            output["logits"].reshape(-1, vocab_size),
            target.reshape(-1),
        )
        total_loss += loss.item()

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate_generation_metrics(
    model, validation_dataset, tokenizer, device, num_examples=30, max_gen_len=300
):
    """Runs greedy decoding and computes assignment metrics on validation examples."""
    model.eval()
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")

    eval_indices = list(range(min(num_examples, len(validation_dataset))))
    target_texts = []
    pred_texts = []

    for idx in eval_indices:
        item = validation_dataset[idx]
        cipher = item["cipher"].unsqueeze(0).to(device)

        # Ground truth plaintext
        if "plain_text" in item:
            ref_text = item["plain_text"]
        else:
            tgt_ids = item["target"].tolist()
            tgt_ids = [t for t in tgt_ids if t not in (0, eos_id)]
            ref_text = tokenizer.decode(tgt_ids)

        # Greedy decoding
        gen_tokens = model.generate(
            cipher=cipher,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            max_length=max_gen_len,
        )

        gen_ids = gen_tokens[0].detach().cpu().tolist()
        if gen_ids and gen_ids[0] == bos_id:
            gen_ids = gen_ids[1:]
        if eos_id in gen_ids:
            gen_ids = gen_ids[: gen_ids.index(eos_id)]

        pred_text = tokenizer.decode(gen_ids)

        target_texts.append(ref_text)
        pred_texts.append(pred_text)

    metrics = compute_all_metrics(target_texts, pred_texts)
    return metrics, target_texts, pred_texts


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Initialize WandB if requested
    wandb_run = None
    if args.wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config=vars(args),
            )
            print("WandB initialized successfully.")
        except Exception as e:
            print(f"Warning: WandB initialization failed ({e}). Proceeding without WandB.")

    # 1. Dataset
    print("\nLoading dataset with aligned truncation (max_cipher_len=%d)..." % args.max_cipher_len)
    full_dataset = BinaryToTextDataset(
        "data/brown_cipher.txt",
        "data/brown_plain.txt",
        max_cipher_len=args.max_cipher_len,
    )
    tokenizer = full_dataset.tokenizer
    vocab_size = tokenizer.get_vocab_size()
    print(f"Total dataset size: {len(full_dataset)} | Vocab size: {vocab_size}")

    train_dataset, val_dataset = split_dataset(
        full_dataset,
        train_ratio=0.9,
        seed=args.seed,
    )
    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

    # 2. DataLoaders
    if args.use_bucketing:
        print("Using LengthBucketBatchSampler for training...")
        train_sampler = LengthBucketBatchSampler(
            dataset=train_dataset,
            batch_sizes=[args.batch_size * 2, args.batch_size, 1, 1],
            boundaries=[2048, 4096, 8192],
            shuffle=True,
            seed=args.seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # 3. Model
    model = BinaryToTextTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        max_cipher_length=args.max_cipher_len,
        max_text_length=args.max_text_len,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (Trainable: {trainable_params:,})")

    # 4. Optimizer, Scheduler, Loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01,
    )

    total_training_steps = (len(train_loader) // args.accum_steps) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_training_steps,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is [PAD]

    # Checkpoint dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    metric_history = {
        "bit_accuracy": [],
        "sequence_accuracy": [],
        "bleu": [],
        "rougeL": [],
    }

    print("\nStarting Training...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*30} Epoch {epoch}/{args.epochs} {'='*30}")

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            accum_steps=args.accum_steps,
            max_batches=args.max_train_batches,
            vocab_size=vocab_size,
            epoch=epoch,
            wandb_run=wandb_run,
        )
        train_losses.append(train_loss)

        # Validation loss
        val_loss = evaluate_loss(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            vocab_size=vocab_size,
            max_batches=args.max_val_batches,
        )
        val_losses.append(val_loss)

        # Greedy decoding evaluation metrics
        print("\nEvaluating greedy generation metrics...")
        metrics, targets, preds = evaluate_generation_metrics(
            model=model,
            validation_dataset=val_dataset,
            tokenizer=tokenizer,
            device=device,
            num_examples=args.eval_samples,
        )

        for k in metric_history:
            metric_history[k].append(metrics.get(k, 0.0))

        print(f"\n--- Epoch {epoch} Results ---")
        print(f"Train Loss:           {train_loss:.4f}")
        print(f"Val Loss:             {val_loss:.4f}")
        print(f"Bit-Level Accuracy:   {metrics['bit_accuracy'] * 100:.2f}%")
        print(f"Sequence Accuracy:    {metrics['sequence_accuracy'] * 100:.2f}%")
        print(f"Levenshtein Distance: {metrics['levenshtein_distance']:.2f}")
        print(f"BLEU Score:           {metrics['bleu'] * 100:.2f}")
        print(f"ROUGE-1 / 2 / L:      {metrics['rouge1']:.4f} / {metrics['rouge2']:.4f} / {metrics['rougeL']:.4f}")

        # Show a sample qualitative output
        if targets and preds:
            print("\n[Sample 1]")
            print(f"TARGET: {targets[0][:120]}...")
            print(f"PRED:   {preds[0][:120]}...")

        # WandB logging
        if wandb_run is not None:
            try:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "val/bit_accuracy": metrics["bit_accuracy"],
                        "val/sequence_accuracy": metrics["sequence_accuracy"],
                        "val/levenshtein_distance": metrics["levenshtein_distance"],
                        "val/bleu": metrics["bleu"],
                        "val/rouge1": metrics["rouge1"],
                        "val/rouge2": metrics["rouge2"],
                        "val/rougeL": metrics["rougeL"],
                    }
                )
            except Exception:
                pass

        # Checkpoint saving
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_path = os.path.join(args.checkpoint_dir, "c1_best.pt")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=val_loss,
                path=best_path,
            )
            print(f"[*] New best model saved to {best_path}")

        epoch_path = os.path.join(args.checkpoint_dir, f"c1_epoch_{epoch}.pt")
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            validation_loss=val_loss,
            path=epoch_path,
        )

    # Plot final curves
    plot_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        metric_history=metric_history,
        output_path="outputs/c1_training_curves.png",
        title="Training Progression - Configuration 1 (Base)",
    )

    if wandb_run is not None:
        wandb_run.finish()

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()