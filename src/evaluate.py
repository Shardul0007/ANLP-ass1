"""
Evaluation script for Configuration 1 (C1: Base Transformer).
Generates predictions using greedy decoding and computes all required assignment metrics:
1. Bit-Level Accuracy (%)
2. Sequence Accuracy (%)
3. Levenshtein Distance
4. BLEU Score
5. ROUGE Scores (ROUGE-1, ROUGE-2, ROUGE-L)
"""

import argparse
import os
from pathlib import Path

import torch

from .dataset import BinaryToTextDataset
from .models.transformer import BinaryToTextTransformer
from .tokenizer import load_tokenizer
from .training import split_dataset
from .utils import compute_all_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate C1 Transformer")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/c1_best.pt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=50,
        help="Number of validation examples to evaluate (or -1 for all)",
    )
    parser.add_argument(
        "--max_cipher_len",
        type=int,
        default=8192,
        help="Max cipher length in bits",
    )
    parser.add_argument(
        "--max_gen_len",
        type=int,
        default=300,
        help="Maximum generation length in tokens",
    )
    parser.add_argument(
        "--d_model", type=int, default=256, help="Model hidden dimension"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--d_ff", type=int, default=1024, help="FFN dimension"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of layers"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs/c1_eval_results.txt",
        help="Path to save evaluation summary",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def detect_dims_from_checkpoint(checkpoint):
    state_dict = checkpoint["model_state_dict"]
    d_model = state_dict["output_projection.weight"].shape[1]
    vocab_size = state_dict["output_projection.weight"].shape[0]
    num_layers = sum(
        1
        for k in state_dict
        if k.startswith("encoder.layers.") and k.endswith(".norm1.gamma")
    )

    d_ff = 1024
    if "encoder.layers.0.ffn.linear1.weight" in state_dict:
        d_ff = state_dict["encoder.layers.0.ffn.linear1.weight"].shape[0]

    max_cipher_len = 8192
    if "encoder_position.positional_encoding" in state_dict:
        max_cipher_len = state_dict[
            "encoder_position.positional_encoding"
        ].shape[1]

    max_text_len = 1024
    if "decoder_position.positional_encoding" in state_dict:
        max_text_len = state_dict[
            "decoder_position.positional_encoding"
        ].shape[1]

    return (
        d_model,
        vocab_size,
        num_layers,
        d_ff,
        max_cipher_len,
        max_text_len,
    )


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = load_tokenizer()
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")

    # Checkpoint path check
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        fallback = "checkpoints/c1_epoch_1.pt"
        if os.path.exists(fallback):
            print(f"Notice: {checkpoint_path} not found. Falling back to {fallback}")
            checkpoint_path = fallback
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Detect dimensions if possible
    try:
        (
            d_model,
            vocab_size,
            num_layers,
            d_ff,
            max_cipher_len,
            max_text_len,
        ) = detect_dims_from_checkpoint(checkpoint)
        print(
            f"Detected from checkpoint: d_model={d_model}, d_ff={d_ff}, "
            f"vocab={vocab_size}, layers={num_layers}, "
            f"max_cipher_len={max_cipher_len}, max_text_len={max_text_len}"
        )
    except Exception:
        d_model = args.d_model
        d_ff = args.d_ff
        vocab_size = tokenizer.get_vocab_size()
        num_layers = args.num_layers
        max_cipher_len = args.max_cipher_len
        max_text_len = args.max_text_len

    # Load validation dataset
    dataset = BinaryToTextDataset(
        "data/brown_cipher.txt",
        "data/brown_plain.txt",
        max_cipher_len=max_cipher_len,
    )

    _, val_dataset = split_dataset(dataset, train_ratio=0.9, seed=args.seed)
    total_val = len(val_dataset)
    eval_count = total_val if args.num_examples < 0 else min(args.num_examples, total_val)

    print(f"Total validation examples: {total_val} | Evaluating: {eval_count}")

    # Initialize model
    model = BinaryToTextTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=args.num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_cipher_length=max_cipher_len,
        max_text_length=max_text_len,
    ).to(device)

    # Load weights with shape-safe filtering
    model_state = model.state_dict()
    filtered_state = {}
    for k, v in checkpoint["model_state_dict"].items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                filtered_state[k] = v
            elif (
                k == "binary_embedding.weight"
                and v.shape[0] == 2
                and model_state[k].shape[0] == 3
                and v.shape[1] == model_state[k].shape[1]
            ):
                model_state[k][:2] = v
                filtered_state[k] = model_state[k]

    model.load_state_dict(filtered_state, strict=False)
    model.eval()

    epoch = checkpoint.get("epoch", "N/A")
    val_loss = checkpoint.get("validation_loss", "N/A")
    print(f"Model loaded (Trained Epoch: {epoch}, Val Loss: {val_loss})")

    targets = []
    predictions = []

    print("\nRunning greedy decoding...")
    for i in range(eval_count):
        item = val_dataset[i]
        cipher = item["cipher"].unsqueeze(0).to(device)

        if "plain_text" in item:
            ref_text = item["plain_text"]
        else:
            tgt_ids = item["target"].tolist()
            tgt_ids = [t for t in tgt_ids if t not in (0, eos_id)]
            ref_text = tokenizer.decode(tgt_ids)

        gen_tokens = model.generate(
            cipher=cipher,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            max_length=args.max_gen_len,
        )

        gen_ids = gen_tokens[0].detach().cpu().tolist()
        if gen_ids and gen_ids[0] == bos_id:
            gen_ids = gen_ids[1:]
        if eos_id in gen_ids:
            gen_ids = gen_ids[: gen_ids.index(eos_id)]

        pred_text = tokenizer.decode(gen_ids)

        targets.append(ref_text)
        predictions.append(pred_text)

        if (i + 1) % 10 == 0 or (i + 1) == eval_count:
            print(f"Processed {i + 1:3d}/{eval_count} examples...")

    # Compute metrics
    print("\nComputing evaluation metrics...")
    metrics = compute_all_metrics(targets, predictions)

    # Print Report
    header = "=" * 70
    output_lines = [
        header,
        "CONFIGURATION C1: BASELINE EVALUATION REPORT",
        header,
        f"Checkpoint:            {checkpoint_path}",
        f"Evaluated Examples:    {eval_count}",
        f"Bit-Level Accuracy:    {metrics['bit_accuracy'] * 100:.2f}%",
        f"Sequence Accuracy:     {metrics['sequence_accuracy'] * 100:.2f}%",
        f"Levenshtein Distance:  {metrics['levenshtein_distance']:.2f}",
        f"BLEU Score:            {metrics['bleu'] * 100:.2f}",
        f"ROUGE-1:               {metrics['rouge1']:.4f}",
        f"ROUGE-2:               {metrics['rouge2']:.4f}",
        f"ROUGE-L:               {metrics['rougeL']:.4f}",
        header,
        "\nQUALITATIVE SAMPLES (First 3 Examples):",
        header,
    ]

    for idx in range(min(3, len(targets))):
        output_lines.extend(
            [
                f"\n[Sample {idx + 1}]",
                f"TARGET:     {targets[idx]}",
                f"PREDICTION: {predictions[idx]}",
                f"Levenshtein Dist: {levenshtein_distance(targets[idx], predictions[idx])}",
            ]
        )

    output_lines.append(header)
    summary_text = "\n".join(output_lines)
    print("\n" + summary_text)

    # Save to outputs
    os.makedirs(Path(args.output_file).parent, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    from .utils import levenshtein_distance
    main()
