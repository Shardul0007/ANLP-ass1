import torch
from torch.utils.data import DataLoader, random_split

from .dataset import (
    BinaryToTextDataset,
    collate_fn,
)
from .models.transformer import (
    BinaryToTextTransformer,
)
from .tokenizer import load_tokenizer


# =========================================
# Configuration
# =========================================

CHECKPOINT = "checkpoints/c1_best.pt"

MAX_CIPHER_LENGTH = 4096

VOCAB_SIZE = 8000

D_MODEL = 256
NUM_HEADS = 8
D_FF = 1024
NUM_LAYERS = 2


# =========================================
# Tokenizer
# =========================================

tokenizer = load_tokenizer()

BOS_ID = tokenizer.token_to_id("[BOS]")
EOS_ID = tokenizer.token_to_id("[EOS]")

print("BOS:", BOS_ID)
print("EOS:", EOS_ID)


# =========================================
# Dataset
# =========================================

dataset = BinaryToTextDataset(
    "data/brown_cipher.txt",
    "data/brown_plain.txt",
)

train_size = int(
    len(dataset) * 0.9
)

validation_size = (
    len(dataset) - train_size
)

generator = torch.Generator().manual_seed(42)

_, validation_dataset = random_split(
    dataset,
    [train_size, validation_size],
    generator=generator,
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


# =========================================
# Load checkpoint
# =========================================

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "Loaded checkpoint from epoch:",
    checkpoint["epoch"],
)

print(
    "Checkpoint validation loss:",
    checkpoint["validation_loss"],
)


# =========================================
# Pick validation example
# =========================================

example = validation_dataset[0]

cipher = example["cipher"].unsqueeze(0)

target = example["target"].unsqueeze(0)

decoder_input = (
    example["decoder_input"]
    .unsqueeze(0)
)


# =========================================
# Original plaintext
# =========================================

target_ids = target[0].tolist()

# Remove padding
target_ids = [
    token_id
    for token_id in target_ids
    if token_id != 0
]

if EOS_ID in target_ids:
    target_ids = target_ids[
        :target_ids.index(EOS_ID)
    ]

original_text = tokenizer.decode(
    target_ids
)


# =========================================
# Cipher
# =========================================

original_cipher_length = cipher.size(1)

cipher = cipher[
    :, :MAX_CIPHER_LENGTH
]

print(
    "\nOriginal cipher length:",
    original_cipher_length,
)

print(
    "Cipher length used:",
    cipher.size(1),
)


# =========================================
# Generate
# =========================================

generated = model.generate(
    cipher,
    bos_token_id=BOS_ID,
    eos_token_id=EOS_ID,
    max_length=200,
)


# =========================================
# Decode prediction
# =========================================

generated_ids = generated[0].tolist()

# Remove BOS
if generated_ids and generated_ids[0] == BOS_ID:
    generated_ids = generated_ids[1:]

# Stop at EOS
if EOS_ID in generated_ids:
    generated_ids = generated_ids[
        :generated_ids.index(EOS_ID)
    ]

generated_text = tokenizer.decode(
    generated_ids
)


# =========================================
# Print
# =========================================

print("\n" + "=" * 70)

print("\nORIGINAL:")
print(original_text)

print("\nGENERATED:")
print(generated_text)

print("\nGenerated token count:")
print(len(generated_ids))

print("\n" + "=" * 70)