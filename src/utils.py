"""
Evaluation metrics and plotting utilities for ANLP Assignment 1.

Metrics required by the assignment:
- Bit-Level Accuracy: The percentage of exact bit matches.
- Sequence Accuracy: The percentage of sequences that are perfectly reconstructed.
- Levenshtein Distance: The edit distance between the target and predicted outputs.
- BLEU and ROUGE Scores: Standard n-gram overlap metrics.
"""

import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Union


# ==============================================================================
# 1. Bit-Level Accuracy
# ==============================================================================

def text_to_bitstring(text: str) -> str:
    """Convert string to binary bit string using UTF-8 byte encoding."""
    encoded_bytes = text.encode("utf-8", errors="replace")
    return "".join(f"{byte:08b}" for byte in encoded_bytes)


def compute_bit_level_accuracy(target_texts: List[str], pred_texts: List[str]) -> float:
    """
    Computes average bit-level accuracy across all target-prediction pairs.
    Each character is converted to its 8-bit representation, and bit-level
    matches are computed against the max bit length of each pair.
    """
    if not target_texts:
        return 0.0

    total_accuracy = 0.0
    for target, pred in zip(target_texts, pred_texts):
        t_bits = text_to_bitstring(target)
        p_bits = text_to_bitstring(pred)

        max_len = max(len(t_bits), len(p_bits))
        if max_len == 0:
            total_accuracy += 1.0
            continue

        min_len = min(len(t_bits), len(p_bits))
        matches = sum(1 for i in range(min_len) if t_bits[i] == p_bits[i])
        total_accuracy += matches / max_len

    return total_accuracy / len(target_texts)


# ==============================================================================
# 2. Sequence Accuracy (Exact Match)
# ==============================================================================

def compute_sequence_accuracy(target_texts: List[str], pred_texts: List[str]) -> float:
    """Percentage of sequences that match exactly."""
    if not target_texts:
        return 0.0
    matches = sum(1 for t, p in zip(target_texts, pred_texts) if t.strip() == p.strip())
    return matches / len(target_texts)


# ==============================================================================
# 3. Levenshtein Distance
# ==============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Computes character-level Levenshtein edit distance between s1 and s2
    using an O(min(len(s1), len(s2))) space dynamic programming algorithm.
    """
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


def compute_levenshtein_distance(target_texts: List[str], pred_texts: List[str]) -> float:
    """Computes average Levenshtein edit distance across all pairs."""
    if not target_texts:
        return 0.0
    total_dist = sum(levenshtein_distance(t, p) for t, p in zip(target_texts, pred_texts))
    return total_dist / len(target_texts)


# ==============================================================================
# 4. BLEU Score
# ==============================================================================

def _get_ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Self-contained smoothed BLEU score for a single sentence."""
    ref_tokens = reference.strip().split()
    hyp_tokens = hypothesis.strip().split()

    if not hyp_tokens:
        return 0.0
    if not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        hyp_ngrams = _get_ngrams(hyp_tokens, n)
        ref_ngrams = _get_ngrams(ref_tokens, n)

        total_hyp = sum(hyp_ngrams.values())
        if total_hyp == 0:
            precisions.append(0.0)
            continue

        clipped = sum(min(count, ref_ngrams[ngram]) for ngram, count in hyp_ngrams.items())
        # Add 0.1 smoothing for zero precision
        precision = (clipped + 0.1) / (total_hyp + 0.1)
        precisions.append(precision)

    log_sum = sum(math.log(p) for p in precisions) / max_n
    geometric_mean = math.exp(log_sum)

    # Brevity penalty
    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)
    if hyp_len > ref_len:
        bp = 1.0
    elif hyp_len == 0:
        bp = 0.0
    else:
        bp = math.exp(1.0 - ref_len / hyp_len)

    return bp * geometric_mean


def compute_bleu(target_texts: List[str], pred_texts: List[str]) -> float:
    """Computes corpus BLEU score across all targets and predictions."""
    try:
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
        references = [[t.strip().split()] for t in target_texts]
        hypotheses = [p.strip().split() for p in pred_texts]
        chencherry = SmoothingFunction()
        return corpus_bleu(references, hypotheses, smoothing_function=chencherry.method1)
    except Exception:
        if not target_texts:
            return 0.0
        scores = [sentence_bleu(t, p) for t, p in zip(target_texts, pred_texts)]
        return sum(scores) / len(scores)


# ==============================================================================
# 5. ROUGE Scores (ROUGE-1, ROUGE-2, ROUGE-L)
# ==============================================================================

def _lcs_length(x: List[str], y: List[str]) -> int:
    """Length of Longest Common Subsequence."""
    m, n = len(x), len(y)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def compute_rouge(
    target_texts: List[str], pred_texts: List[str]
) -> Dict[str, float]:
    """
    Computes ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.
    Uses `rouge_score` if installed, otherwise falls back to a clean native implementation.
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1, r2, rl = 0.0, 0.0, 0.0
        n = len(target_texts)
        if n == 0:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        for t, p in zip(target_texts, pred_texts):
            scores = scorer.score(t, p)
            r1 += scores["rouge1"].fmeasure
            r2 += scores["rouge2"].fmeasure
            rl += scores["rougeL"].fmeasure

        return {
            "rouge1": r1 / n,
            "rouge2": r2 / n,
            "rougeL": rl / n,
        }
    except Exception:
        # Fallback pure python calculation
        if not target_texts:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        r1_total, r2_total, rl_total = 0.0, 0.0, 0.0
        for target, pred in zip(target_texts, pred_texts):
            t_toks = target.strip().split()
            p_toks = pred.strip().split()

            # ROUGE-1
            t_cnt1 = Counter(t_toks)
            p_cnt1 = Counter(p_toks)
            overlap1 = sum(min(p_cnt1[w], t_cnt1[w]) for w in p_cnt1)
            p1 = overlap1 / len(p_toks) if p_toks else 0.0
            r1 = overlap1 / len(t_toks) if t_toks else 0.0
            f1 = (2 * p1 * r1) / (p1 + r1) if (p1 + r1) > 0 else 0.0
            r1_total += f1

            # ROUGE-2
            t_ng2 = _get_ngrams(t_toks, 2)
            p_ng2 = _get_ngrams(p_toks, 2)
            overlap2 = sum(min(p_ng2[ng], t_ng2[ng]) for ng in p_ng2)
            p2 = overlap2 / sum(p_ng2.values()) if sum(p_ng2.values()) else 0.0
            r2 = overlap2 / sum(t_ng2.values()) if sum(t_ng2.values()) else 0.0
            f2 = (2 * p2 * r2) / (p2 + r2) if (p2 + r2) > 0 else 0.0
            r2_total += f2

            # ROUGE-L
            lcs = _lcs_length(t_toks, p_toks)
            pl = lcs / len(p_toks) if p_toks else 0.0
            rl = lcs / len(t_toks) if t_toks else 0.0
            fl = (2 * pl * rl) / (pl + rl) if (pl + rl) > 0 else 0.0
            rl_total += fl

        n = len(target_texts)
        return {
            "rouge1": r1_total / n,
            "rouge2": r2_total / n,
            "rougeL": rl_total / n,
        }


# ==============================================================================
# 6. Combined Evaluation Function
# ==============================================================================

def compute_all_metrics(
    target_texts: List[str], pred_texts: List[str]
) -> Dict[str, float]:
    """Computes all five metrics required by the assignment."""
    bit_acc = compute_bit_level_accuracy(target_texts, pred_texts)
    seq_acc = compute_sequence_accuracy(target_texts, pred_texts)
    lev_dist = compute_levenshtein_distance(target_texts, pred_texts)
    bleu = compute_bleu(target_texts, pred_texts)
    rouge = compute_rouge(target_texts, pred_texts)

    return {
        "bit_accuracy": bit_acc,
        "sequence_accuracy": seq_acc,
        "levenshtein_distance": lev_dist,
        "bleu": bleu,
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
    }


# ==============================================================================
# 7. Plotting Utilities
# ==============================================================================

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    metric_history: Dict[str, List[float]] = None,
    output_path: str = "outputs/training_curves.png",
    title: str = "Training Progress (Configuration C1)",
):
    """Saves training and validation loss curves and metrics to file."""
    try:
        import matplotlib.pyplot as plt

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        epochs = list(range(1, len(train_losses) + 1))

        if metric_history:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

        ax1.plot(epochs, train_losses, "b-o", label="Train Loss")
        ax1.plot(epochs, val_losses, "r-s", label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("CrossEntropy Loss")
        ax1.set_title("Loss Curves")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        if metric_history:
            for metric_name, values in metric_history.items():
                ax2.plot(epochs[: len(values)], values, "-o", label=metric_name)
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Score")
            ax2.set_title("Evaluation Metrics")
            ax2.grid(True, linestyle="--", alpha=0.6)
            ax2.legend()

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"Plot saved to: {output_path}")
    except Exception as e:
        print(f"Could not generate plot ({e}). Matplotlib might not be available.")
