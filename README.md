# Cipher Transformer Ablation Study & Byte Latent Transformers (BLT)

**Course**: Advanced Natural Language Processing (Spring 2026)  
**Assignment 1**: Custom Transformers from Scratch, Architectural Variants & BLT  

---

## 1. Project Overview & Architecture

This repository contains a full Seq2Seq Transformer built entirely from scratch in PyTorch (without using `nn.Transformer` or `nn.MultiheadAttention`) to decrypt binary-encoded substitution ciphers into plaintext. A systematic 5-model controlled ablation study is conducted to evaluate the isolated effects of Positional Encodings (Sinusoidal vs. RoPE), Attention Mechanisms (MHA vs. GQA), Layer Normalization (Pre-LN vs. RMSNorm), and Tokenization (Custom Subword BPE vs. Token-Free BLT).

### Architectural Configurations (C1 – C5)

All models share consistent core hyperparameters: `d_model = 256`, `num_heads = 8`, `num_encoder_layers = 4`, `num_decoder_layers = 4`, `d_ff = 1024`, `dropout = 0.1`, `batch_size = 64`, `learning_rate = 3e-4` with 2,000-step linear warmup and cosine decay over 50 epochs.

| Config | Change from Base | Positional Encoding | Attention Mechanism | Normalization | Tokenization | Total Parameters |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **C1** | **None (Base)** | Sinusoidal Absolute | Multi-Head Attention (MHA) | Pre-LayerNorm | Custom Subword (BPE) | 10,808,082 |
| **C2** | **Positional Encoding** | **RoPE (Rotary)** | Multi-Head Attention (MHA) | Pre-LayerNorm | Custom Subword (BPE) | 10,808,082 |
| **C3** | **Attention Mechanism** | Sinusoidal Absolute | **Grouped-Query Attention (GQA)** | Pre-LayerNorm | Custom Subword (BPE) | **9,623,826** |
| **C4** | **Normalization** | Sinusoidal Absolute | Multi-Head Attention (MHA) | **RMSNorm** | Custom Subword (BPE) | 10,802,450 |
| **C5** | **Tokenization** | Sinusoidal Absolute | Multi-Head Attention (MHA) | Pre-LayerNorm | **BLT (Token-Free)** | **8,430,083** |

---

## 2. Experimental Results & Ablation Analysis

### Summary Table

| Configuration | Bit-Level Acc | Sequence Acc | Normalized Levenshtein | BLEU Score | ROUGE-L | Best Val Loss | Epoch Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **C1 (Base Pre-LN)** | 90.74% | 25.25% | 0.0259 | 82.75 | 0.9241 | 1.5749 | ~83.5s |
| **C2 (RoPE)** | **91.14%** | 24.71% | 0.0262 | 82.74 | 0.9239 | 1.5775 | ~84.6s |
| **C3 (GQA)** | 89.35% | 21.41% | 0.0302 | 80.86 | 0.9149 | 1.6223 | ~84.4s |
| **C4 (RMSNorm)** | 90.65% | **25.75%** | **0.0261** | **82.80** | 0.9237 | **1.5738** | ~84.8s |
| **C5 (BLT Token-Free)** | 68.25% | 0.00% | 0.7598 | *N/A (Byte-level)* | *N/A (Byte-level)* | 2.0555 | **~58.2s** |
| *Naive Baseline A (Freq Byte)* | 66.14% | 0.00% | 0.8317 | — | — | — | — |
| *Naive Baseline B (Unigram)* | 67.40% | 0.00% | 0.8362 | — | — | — | — |

### Resource Utilization & Efficiency

| Configuration | Tokens / Sec | Bytes / Sec | Peak VRAM (MB) | Epoch Time (s) | Parameters |
|:---|:---:|:---:|:---:|:---:|:---:|
| **C1** | 5,824.5 | 48,210.2 | 1420.5 | 83.5s | 10.81M |
| **C2** | 5,642.1 | 47,120.8 | 1435.2 | 84.6s | 10.81M |
| **C3** | 6,120.4 | 51,040.1 | **1180.4** | 84.4s | **9.62M (-11%)** |
| **C4** | 6,010.8 | 49,820.3 | 1412.0 | 84.8s | 10.80M |
| **C5 (BLT)** | 43,048.5 | **430,484.9** | **920.0** | **58.2s (-30%)** | **8.43M (-22%)** |

---

## 3. Weights & Biases (WandB) & Hugging Face

### WandB Dashboard
* **WandB Project**: [https://wandb.ai/shardul0750-iiit-hyderabad/ANLP_A1](https://wandb.ai/shardul0750-iiit-hyderabad/ANLP_A1)
* **Run C1 (Base)**: [https://wandb.ai/shardul0750-iiit-hyderabad/ANLP_A1/runs/dnkcbcev](https://wandb.ai/shardul0750-iiit-hyderabad/ANLP_A1/runs/dnkcbcev)
* **Run C2 (RoPE)**: [https://wandb.ai/shardul0750-iiit-hyderabad/ANLP_A1/runs/w6wjcy0h](https://wandb.ai/shardul0750-iiit-hyderabad/ANLP_A1/runs/w6wjcy0h)
* **Run C3 (GQA)**: [https://wandb.ai/shardul0750-iiit-hyderabad/ANLP_A1/runs/jnso2i03](https://wandb.ai/shardul0750-iiit-hyderabad/ANLP_A1/runs/jnso2i03)
* **Run C4 (RMSNorm)**: [https://wandb.ai/shardul0750-iiit-hyderabad/ANLP_A1/runs/7xrgmm0l](https://wandb.ai/shardul0750-iiit-hyderabad/ANLP_A1/runs/7xrgmm0l)
* **Run C5 (BLT)**: [https://wandb.ai/shardul0750-iiit-hyderabad/ANLP_A1/runs/ygzpumov](https://wandb.ai/shardul0750-iiit-hyderabad/ANLP_A1/runs/ygzpumov)

### Hugging Face Checkpoints
All 5 model checkpoints and evaluation artifacts are live on Hugging Face Hub:
* **Model Hub**: [https://huggingface.co/Shardul007/ANLP-A1-Checkpoints](https://huggingface.co/Shardul007/ANLP-A1-Checkpoints)

To download or load checkpoints in Python:
```python
from huggingface_hub import hf_hub_download
import torch

ckpt_path = hf_hub_download(
    repo_id="Shardul007/ANLP-A1-Checkpoints",
    filename="c1/best_model.pt"
)
checkpoint = torch.load(ckpt_path, map_location="cpu")
print("Loaded C1 checkpoint successfully!")
```

---

## 4. Directory Structure

```text
2024101077_assignment1/
|-- src/
|   |-- models/
|   |   |-- __init__.py      # Pre-LN Encoder/Decoder, Seq2SeqTransformer
|   |   |-- attention.py     # Custom Scaled Dot-Product, MHA and GQA
|   |   |-- positional.py    # Sinusoidal Positional Encoding and RoPE
|   |   |-- norm.py          # Pre-LayerNorm and RMSNorm from scratch
|   |   `-- blt.py           # Byte Latent Transformer (Local Patch Encoder/Decoder)
|   |-- dataset.py           # Aligned chunking, tokenized & token-free loaders
|   |-- tokenizer.py         # Custom from-scratch Byte-Pair Encoding (BPE)
|   |-- train.py             # Main training loop with WandB & AdamW
|   |-- evaluate.py          # Standalone evaluation & greedy decoding
|   `-- utils.py             # Metrics (Bit Acc, BLEU, ROUGE, Lev) & plots
|-- outputs/                 # Saved learning curves, plots, and metric logs
|   |-- ablation_learning_curves.png
|   |-- metrics_comparison.png
|   |-- c5_vs_c1.png
|   |-- metrics_c1.json ... metrics_c5.json
|   `-- losses_c1.json ... losses_c5.json
|-- requirements.txt         # Dependencies
|-- main.py                  # Ablation study runner & report generator
|-- README.md                # Reproduction guide & links
`-- Report.pdf               # 6-page comprehensive technical report
```

---

## 5. Reproduction Instructions

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Run Individual Configurations
```bash
# C1: Base Pre-LN Transformer
python src/train.py --run_name c1

# C2: Rotary Position Embeddings (RoPE)
python src/train.py --run_name c2 --positional_encoding rope

# C3: Grouped-Query Attention (GQA)
python src/train.py --run_name c3 --attention_type gqa

# C4: RMSNorm
python src/train.py --run_name c4 --norm_type rmsnorm

# C5: Byte Latent Transformer (BLT)
python src/train.py --run_name c5 --tokenization blt
```

### 3. Generate Complete Plots & Summary
```bash
python main.py
```
