"""
Standalone Evaluation script for trained models (C1–C5).
Loads a saved checkpoint, runs greedy autoregressive decoding on the test set,
and computes all required assignment metrics.
"""

import argparse
import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import (
    build_dataloaders,
    BYTE_PAD,
    PLAIN_CHUNK_SIZE,
    DATASET_DIR,
)
from src.models import Seq2SeqTransformer
from src.models.blt import BLTSeq2SeqModel
from src.utils import compute_all_metrics, compute_naive_baselines


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Cipher Transformer")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/c1/best_model.pt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="c1",
        help="Run name / config tag",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of test samples to evaluate (default: all)",
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")

    print(f"Loading checkpoint from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    tokenization = config.get("tokenization", "subword")
    is_blt = tokenization == "blt"

    print(f"Config: {config.get('run_name', args.run_name)}")
    print(f"Tokenization: {tokenization}")

    # Build dataset
    data_info = build_dataloaders(
        tokenization=tokenization,
        batch_size=config.get("batch_size", 64),
        max_seq_len=config.get("max_seq_len", 512),
        vocab_size=config.get("bpe_vocab_size", 8000),
        seed=config.get("seed", 42),
        data_dir=DATASET_DIR,
    )
    test_loader = data_info["test_loader"]

    # Build model
    if is_blt:
        model = BLTSeq2SeqModel(
            d_model=config.get("d_model", 256),
            num_heads=config.get("num_heads", 8),
            num_encoder_layers=config.get("num_encoder_layers", 4),
            num_decoder_layers=config.get("num_decoder_layers", 4),
            d_ff=config.get("d_ff", 1024),
            dropout=config.get("dropout", 0.1),
            max_seq_len=config.get("max_seq_len", 512),
            patch_size=config.get("blt_patch_size", 9),
            d_local=config.get("d_model", 256),
            local_heads=config.get("num_heads", 8),
        )
    else:
        model = Seq2SeqTransformer(
            src_vocab_size=data_info["src_vocab_size"],
            tgt_vocab_size=data_info["tgt_vocab_size"],
            d_model=config.get("d_model", 256),
            num_heads=config.get("num_heads", 8),
            num_encoder_layers=config.get("num_encoder_layers", 4),
            num_decoder_layers=config.get("num_decoder_layers", 4),
            d_ff=config.get("d_ff", 1024),
            dropout=config.get("dropout", 0.1),
            max_seq_len=config.get("max_seq_len", 512),
            pad_idx=data_info["pad_idx"],
            attention_type=config.get("attention_type", "mha"),
            norm_type=config.get("norm_type", "layernorm"),
            positional_encoding=config.get("positional_encoding", "sinusoidal"),
            num_kv_heads=config.get("gqa_kv_heads", 2),
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print("\nRunning greedy decoding on test set...")
    predictions = []
    targets = []
    tokenizer_tgt = data_info.get("tokenizer_tgt")

    with torch.no_grad():
        for batch in test_loader:
            src, tgt = batch
            src = src.to(device)
            tgt = tgt.to(device)

            if is_blt:
                pred_ids = model.greedy_decode(src, max_len=PLAIN_CHUNK_SIZE + 16)
            else:
                pred_ids = model.greedy_decode(
                    src,
                    bos_idx=data_info["bos_idx"],
                    eos_idx=data_info["eos_idx"],
                    max_len=config.get("max_seq_len", 512),
                )

            for i in range(pred_ids.size(0)):
                pred_seq = pred_ids[i].cpu().tolist()
                tgt_seq = tgt[i].cpu().tolist()

                if is_blt:
                    pred_str = bytes([b for b in pred_seq if b < 256 and b != BYTE_PAD]).decode(
                        "utf-8", errors="replace"
                    )
                    tgt_str = bytes([b for b in tgt_seq if b < 256 and b != BYTE_PAD]).decode(
                        "utf-8", errors="replace"
                    )
                else:
                    eos_id = data_info["eos_idx"]
                    pad_id = data_info["pad_idx"]
                    bos_id = data_info["bos_idx"]

                    if eos_id in pred_seq:
                        pred_seq = pred_seq[: pred_seq.index(eos_id)]
                    pred_seq = [t for t in pred_seq if t not in (pad_id, bos_id, eos_id)]

                    if eos_id in tgt_seq:
                        tgt_seq = tgt_seq[: tgt_seq.index(eos_id)]
                    tgt_seq = [t for t in tgt_seq if t not in (pad_id, bos_id, eos_id)]

                    pred_str = tokenizer_tgt.decode(pred_seq) if pred_seq else ""
                    tgt_str = tokenizer_tgt.decode(tgt_seq) if tgt_seq else ""

                predictions.append(pred_str)
                targets.append(tgt_str)

                if args.num_samples and len(predictions) >= args.num_samples:
                    break
            if args.num_samples and len(predictions) >= args.num_samples:
                break

    metrics = compute_all_metrics(predictions, targets, is_token_free=is_blt)

    train_targets = data_info["splits"]["train"]["plain"]
    test_targets = data_info["splits"]["test"]["plain"]
    baselines = compute_naive_baselines(train_targets, test_targets)

    print("\n" + "=" * 60)
    print(f"EVALUATION REPORT — {args.run_name.upper()}")
    print("=" * 60)
    print(f"Evaluated Samples:     {len(predictions)}")
    print(f"Bit-Level Accuracy:    {metrics.get('bit_accuracy', 0.0) * 100:.2f}%")
    print(f"Sequence Accuracy:     {metrics.get('sequence_accuracy', 0.0) * 100:.2f}%")
    print(f"Levenshtein (Norm):    {metrics.get('levenshtein_normalized', 0.0):.4f}")
    print(f"Levenshtein (Raw):     {metrics.get('levenshtein_raw', 0.0):.2f}")
    if not is_blt:
        print(f"BLEU Score:            {metrics.get('bleu', 0.0):.2f}")
        print(f"ROUGE-1:               {metrics.get('rouge1', 0.0):.4f}")
        print(f"ROUGE-2:               {metrics.get('rouge2', 0.0):.4f}")
        print(f"ROUGE-L:               {metrics.get('rougeL', 0.0):.4f}")
    print("=" * 60)

    if baselines:
        print("\nNAIVE BASELINES:")
        if "baseline_a" in baselines:
            ba = baselines["baseline_a"]
            print(f"  Most Frequent Byte: Bit Acc = {ba['bit_accuracy'] * 100:.2f}%, Seq Acc = {ba['sequence_accuracy'] * 100:.2f}%")
        if "baseline_b" in baselines:
            bb = baselines["baseline_b"]
            print(f"  Unigram Sample:     Bit Acc = {bb['bit_accuracy'] * 100:.2f}%, Seq Acc = {bb['sequence_accuracy'] * 100:.2f}%")

    print("\nSAMPLE PREDICTIONS (First 3):")
    for idx in range(min(3, len(targets))):
        print(f"\n[Sample {idx + 1}]")
        print(f"TARGET:     {targets[idx]}")
        print(f"PREDICTION: {predictions[idx]}")


if __name__ == "__main__":
    main()
