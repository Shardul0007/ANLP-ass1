import os
import json
import collections
import random
import math
from typing import Optional

import torch
import numpy as np

# Optional wandb
try:
    import wandb
except ImportError:
    wandb = None

# Optional Hugging Face Hub
try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    HfApi = None
    hf_hub_download = None

# Optional Levenshtein
try:
    import Levenshtein
except ImportError:
    Levenshtein = None

# Optional sacrebleu
try:
    import sacrebleu
except ImportError:
    sacrebleu = None

# Optional rouge_score
try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None


# ==============================================================================
# W&B Helpers
# ==============================================================================

def init_wandb(project: str, config: dict, name: str | None = None):
    if wandb is not None:
        try:
            return wandb.init(project=project, config=config, name=name)
        except Exception as e:
            print(f"Warning: WandB initialization failed ({e}). Proceeding without WandB.")
    class DummyRun:
        url = "N/A (WandB disabled/unavailable)"
    return DummyRun()


def log_wandb(metrics: dict, step: int | None = None) -> None:
    if wandb is not None and wandb.run is not None:
        try:
            wandb.log(metrics, step=step)
        except Exception:
            pass


def finish_wandb() -> None:
    if wandb is not None and wandb.run is not None:
        try:
            wandb.finish()
        except Exception:
            pass


# ==============================================================================
# HF Hub Helpers
# ==============================================================================

def push_to_hub(
    path: str,
    repo_id: str,
    path_in_repo: str | None = None,
    token: str | None = None,
) -> str:
    token = token or os.environ.get("HF_TOKEN")
    if HfApi is None:
        raise ImportError("huggingface_hub is not installed.")
    api = HfApi()
    api.create_repo(repo_id=repo_id, token=token, exist_ok=True)
    return api.upload_file(
        path_or_fileobj=path,
        path_in_repo=path_in_repo or os.path.basename(path),
        repo_id=repo_id,
        token=token,
    )


def push_folder_to_hub(
    folder_path: str,
    repo_id: str,
    token: str | None = None,
) -> str:
    token = token or os.environ.get("HF_TOKEN")
    if HfApi is None:
        raise ImportError("huggingface_hub is not installed.")
    api = HfApi()
    api.create_repo(repo_id=repo_id, token=token, exist_ok=True)
    return api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        token=token,
    )


def pull_from_hub(
    repo_id: str,
    filename: str,
    local_dir: str = "checkpoints",
    token: str | None = None,
) -> str:
    token = token or os.environ.get("HF_TOKEN")
    if hf_hub_download is None:
        raise ImportError("huggingface_hub is not installed.")
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        token=token,
    )


def save_and_push(
    model: torch.nn.Module,
    repo_id: str,
    filename: str = "model.pt",
    local_dir: str = "checkpoints",
    token: str | None = None,
) -> str:
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)
    torch.save(model.state_dict(), local_path)
    return push_to_hub(local_path, repo_id, filename, token)


def load_from_hub(
    model: torch.nn.Module,
    repo_id: str,
    filename: str = "model.pt",
    local_dir: str = "checkpoints",
    device: str = "cpu",
    token: str | None = None,
) -> torch.nn.Module:
    path = pull_from_hub(repo_id, filename, local_dir, token)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model


# ==============================================================================
# Evaluation Metrics
# ==============================================================================

def _str_to_bytes(s: str) -> bytes:
    if isinstance(s, bytes):
        return s
    return s.encode("utf-8")


def _bytes_to_bits(b: bytes) -> list[int]:
    bits = []
    for byte in b:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def bit_level_accuracy(predictions: list[str], targets: list[str]) -> float:
    """Bit-Level Accuracy.
    Convert both to raw bytes, right-pad shorter with zero bytes,
    expand each byte to 8 bits, and report % matching bits.
    """
    total_bits = 0
    matching_bits = 0

    for pred, tgt in zip(predictions, targets):
        pred_bytes = _str_to_bytes(pred)
        tgt_bytes = _str_to_bytes(tgt)

        max_len = max(len(pred_bytes), len(tgt_bytes))
        pred_bytes = pred_bytes.ljust(max_len, b"\x00")
        tgt_bytes = tgt_bytes.ljust(max_len, b"\x00")

        pred_bits = _bytes_to_bits(pred_bytes)
        tgt_bits = _bytes_to_bits(tgt_bytes)

        total_bits += len(pred_bits)
        matching_bits += sum(p == t for p, t in zip(pred_bits, tgt_bits))

    return matching_bits / total_bits if total_bits > 0 else 0.0


def sequence_accuracy(predictions: list[str], targets: list[str]) -> float:
    """Sequence Accuracy: % of pairs that are an exact match."""
    if not predictions:
        return 0.0
    exact = sum(1 for p, t in zip(predictions, targets) if p == t)
    return exact / len(predictions)


def _native_levenshtein(s1, s2) -> int:
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1] + [0] * len(s2)
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row[j + 1] = min(insertions, deletions, substitutions)
        previous_row = current_row
    return previous_row[-1]


def levenshtein_metrics(
    predictions: list[str],
    targets: list[str],
    byte_level: bool = False,
) -> dict:
    """Levenshtein Distance metrics (raw and length-normalized)."""
    total_raw = 0.0
    total_norm = 0.0

    for pred, tgt in zip(predictions, targets):
        if byte_level:
            pred_seq = list(_str_to_bytes(pred))
            tgt_seq = list(_str_to_bytes(tgt))
        else:
            pred_seq = pred
            tgt_seq = tgt

        if Levenshtein is not None and not byte_level:
            dist = Levenshtein.distance(pred_seq, tgt_seq)
        else:
            dist = _native_levenshtein(pred_seq, tgt_seq)

        total_raw += dist
        max_len = max(len(pred_seq), len(tgt_seq))
        total_norm += (dist / max_len) if max_len > 0 else 0.0

    n = len(predictions) if predictions else 1
    return {
        "levenshtein_raw": total_raw / n,
        "levenshtein_normalized": total_norm / n,
    }


def compute_bleu(predictions: list[str], targets: list[str]) -> float:
    """BLEU score using sacrebleu with native fallback."""
    if sacrebleu is not None:
        bleu = sacrebleu.corpus_bleu(predictions, [targets])
        return bleu.score

    # Native BLEU fallback
    if not predictions:
        return 0.0
    scores = []
    for hyp, ref in zip(predictions, targets):
        hyp_toks = hyp.strip().split()
        ref_toks = ref.strip().split()
        if not hyp_toks or not ref_toks:
            scores.append(0.0)
            continue
        precisions = []
        for n in range(1, 5):
            hyp_ng = collections.Counter(tuple(hyp_toks[i : i + n]) for i in range(len(hyp_toks) - n + 1))
            ref_ng = collections.Counter(tuple(ref_toks[i : i + n]) for i in range(len(ref_toks) - n + 1))
            tot = sum(hyp_ng.values())
            if tot == 0:
                precisions.append(1e-8)
            else:
                clipped = sum(min(cnt, ref_ng[ng]) for ng, cnt in hyp_ng.items())
                precisions.append((clipped + 0.1) / (tot + 0.1))
        geom = math.exp(sum(math.log(p) for p in precisions) / 4)
        bp = 1.0 if len(hyp_toks) > len(ref_toks) else math.exp(1.0 - len(ref_toks) / len(hyp_toks))
        scores.append(bp * geom * 100)
    return sum(scores) / len(scores) if scores else 0.0


def compute_rouge(predictions: list[str], targets: list[str]) -> dict:
    """ROUGE-1/2/L using rouge-score with native fallback."""
    if rouge_scorer is not None:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1_scores, r2_scores, rl_scores = [], [], []
        for pred, tgt in zip(predictions, targets):
            scores = scorer.score(tgt, pred)
            r1_scores.append(scores["rouge1"].fmeasure)
            r2_scores.append(scores["rouge2"].fmeasure)
            rl_scores.append(scores["rougeL"].fmeasure)

        return {
            "rouge1": sum(r1_scores) / len(r1_scores) if r1_scores else 0.0,
            "rouge2": sum(r2_scores) / len(r2_scores) if r2_scores else 0.0,
            "rougeL": sum(rl_scores) / len(rl_scores) if rl_scores else 0.0,
        }

    # Native ROUGE fallback
    if not predictions:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    r1_total, r2_total, rl_total = 0.0, 0.0, 0.0
    for pred, target in zip(predictions, targets):
        t_toks = target.strip().split()
        p_toks = pred.strip().split()

        t_cnt1 = collections.Counter(t_toks)
        p_cnt1 = collections.Counter(p_toks)
        ov1 = sum(min(p_cnt1[w], t_cnt1[w]) for w in p_cnt1)
        p1 = ov1 / len(p_toks) if p_toks else 0.0
        r1 = ov1 / len(t_toks) if t_toks else 0.0
        r1_total += (2 * p1 * r1) / (p1 + r1) if (p1 + r1) > 0 else 0.0

        t_ng2 = collections.Counter(tuple(t_toks[i : i + 2]) for i in range(len(t_toks) - 1))
        p_ng2 = collections.Counter(tuple(p_toks[i : i + 2]) for i in range(len(p_toks) - 1))
        ov2 = sum(min(p_ng2[ng], t_ng2[ng]) for ng in p_ng2)
        p2 = ov2 / sum(p_ng2.values()) if sum(p_ng2.values()) else 0.0
        r2 = ov2 / sum(t_ng2.values()) if sum(t_ng2.values()) else 0.0
        r2_total += (2 * p2 * r2) / (p2 + r2) if (p2 + r2) > 0 else 0.0

        # LCS for ROUGE-L
        m, n = len(t_toks), len(p_toks)
        prev = [0] * (n + 1)
        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            for j in range(1, n + 1):
                if t_toks[i - 1] == p_toks[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev = curr
        lcs = prev[-1]
        pl = lcs / len(p_toks) if p_toks else 0.0
        rl = lcs / len(t_toks) if t_toks else 0.0
        rl_total += (2 * pl * rl) / (pl + rl) if (pl + rl) > 0 else 0.0

    n_samples = len(predictions)
    return {
        "rouge1": r1_total / n_samples,
        "rouge2": r2_total / n_samples,
        "rougeL": rl_total / n_samples,
    }


def compute_naive_baselines(train_targets: list[str], test_targets: list[str]) -> dict:
    """Compute baselines on raw text before evaluating."""
    if not train_targets or not test_targets:
        return {}

    counter = collections.Counter()
    for tgt in train_targets:
        for token in tgt:
            counter[token] += 1

    if not counter:
        return {}

    most_freq_token = counter.most_common(1)[0][0]

    # Baseline A: Most frequent byte
    preds_a = []
    for tgt in test_targets:
        preds_a.append(most_freq_token * len(tgt))

    # Baseline B: Unigram sample
    population = list(counter.keys())
    weights = list(counter.values())
    preds_b = []
    for tgt in test_targets:
        sampled = random.choices(population, weights=weights, k=len(tgt))
        preds_b.append("".join(sampled))

    metrics_a = {
        "bit_accuracy": bit_level_accuracy(preds_a, test_targets),
        "sequence_accuracy": sequence_accuracy(preds_a, test_targets),
        **levenshtein_metrics(preds_a, test_targets, byte_level=False),
    }

    metrics_b = {
        "bit_accuracy": bit_level_accuracy(preds_b, test_targets),
        "sequence_accuracy": sequence_accuracy(preds_b, test_targets),
        **levenshtein_metrics(preds_b, test_targets, byte_level=False),
    }

    return {"baseline_a": metrics_a, "baseline_b": metrics_b}


def compute_all_metrics(
    predictions: list[str],
    targets: list[str],
    is_token_free: bool = False,
) -> dict:
    """Compute all evaluation metrics for a set of predictions."""
    metrics = {
        "bit_accuracy": bit_level_accuracy(predictions, targets),
        "sequence_accuracy": sequence_accuracy(predictions, targets),
        **levenshtein_metrics(predictions, targets, byte_level=is_token_free),
    }

    if is_token_free:
        metrics["bleu"] = "N/A - token-free"
        metrics["rouge1"] = "N/A - token-free"
        metrics["rouge2"] = "N/A - token-free"
        metrics["rougeL"] = "N/A - token-free"
    else:
        metrics["bleu"] = compute_bleu(predictions, targets)
        metrics.update(compute_rouge(predictions, targets))

    return metrics


def save_metrics_json(metrics: dict, run_name: str, output_dir: str = "outputs") -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"metrics_{run_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return out_path


# ==============================================================================
# Plotting Utilities
# ==============================================================================

def plot_metrics_comparison(results: dict, output_dir: str = "outputs"):
    """Grouped bar chart comparing all 5 configurations."""
    import matplotlib.pyplot as plt

    configs = [c for c in ["C1", "C2", "C3", "C4", "C5"] if c in results]
    if not configs:
        return

    metric_keys = ["bit_accuracy", "sequence_accuracy", "levenshtein_normalized"]
    metric_labels = ["Bit-Level Acc", "Seq Acc", "Norm Levenshtein"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(configs))
    width = 0.25

    for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        vals = [results[c].get(key, 0.0) for c in configs]
        ax.bar(x + i * width, vals, width, label=label)

    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: Metrics Across Configurations")
    ax.set_xticks(x + width)
    ax.set_xticklabels(configs)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "metrics_comparison.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Metrics comparison plot saved to: {plot_path}")


def plot_c5_vs_c1(
    c1_metrics: dict,
    c5_metrics: dict,
    c1_speed: dict,
    c5_speed: dict,
    output_dir: str = "outputs",
):
    """Plot direct C1 vs C5 comparison."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Performance
    metrics = ["Bit Acc", "Seq Acc", "Norm Lev"]
    c1_vals = [
        c1_metrics.get("bit_accuracy", 0.0),
        c1_metrics.get("sequence_accuracy", 0.0),
        c1_metrics.get("levenshtein_normalized", 0.0),
    ]
    c5_vals = [
        c5_metrics.get("bit_accuracy", 0.0),
        c5_metrics.get("sequence_accuracy", 0.0),
        c5_metrics.get("levenshtein_normalized", 0.0),
    ]

    x = np.arange(len(metrics))
    width = 0.35
    ax1.bar(x - width / 2, c1_vals, width, label="C1 (Tokenized)", color="steelblue")
    ax1.bar(x + width / 2, c5_vals, width, label="C5 (BLT)", color="darkorange")
    ax1.set_title("Task Performance: C1 vs C5")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Speed & Throughput
    speed_labels = ["Tokens/s (k)", "Bytes/s (k)"]
    c1_sp = [c1_speed.get("tokens_per_sec", 0.0) / 1000, c1_speed.get("bytes_per_sec", 0.0) / 1000]
    c5_sp = [c5_speed.get("tokens_per_sec", 0.0) / 1000, c5_speed.get("bytes_per_sec", 0.0) / 1000]

    x2 = np.arange(len(speed_labels))
    ax2.bar(x2 - width / 2, c1_sp, width, label="C1 (Tokenized)", color="steelblue")
    ax2.bar(x2 + width / 2, c5_sp, width, label="C5 (BLT)", color="darkorange")
    ax2.set_title("Throughput: C1 vs C5")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(speed_labels)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "c5_vs_c1.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"C5 vs C1 comparison plot saved to: {plot_path}")
