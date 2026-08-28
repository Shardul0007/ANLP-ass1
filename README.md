# ANLP Transformer Assignment

This repository contains the current implementation for an ANLP assignment on
decoding binary cipher sequences into plaintext. The model is a PyTorch
encoder-decoder Transformer with custom attention, masking, normalization,
feed-forward, positional encoding, and training components.

## Project layout

- `src/models/`: Transformer encoder, decoder, attention, masks, and supporting layers
- `src/dataset.py`: paired binary-cipher/plaintext dataset and batch collation
- `src/tokenizer.py`: tokenizer loading and related utilities
- `src/train.py`: configured training entry point
- `src/evaluate.py`: checkpoint-based validation example and generation
- `src/benchmark.py`, `src/benchmark_train.py`: benchmarking entry points
- `src/analyse_dataset.py`, `src/analyse_lengths.py`: dataset analysis scripts
- `data/`: cipher/plaintext pairs and the tokenizer JSON used by the dataset

## Running the code

Run commands from the repository root. Install the Python dependencies in the
target environment first, including PyTorch and the `tokenizers` package.

```powershell
python -m src.train
python -m src.evaluate
```

`src/evaluate.py` expects `checkpoints/c1_epoch_1.pt` to exist. Generated
checkpoints are ignored by Git because the current checkpoint is approximately
38 MB; keep or transfer such files separately, or use Git LFS if they need to
be versioned later.

This README intentionally does not report training or evaluation results.