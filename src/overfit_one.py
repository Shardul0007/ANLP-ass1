import torch
import torch.nn as nn

from .dataset import BinaryToTextDataset
from .models.masks import create_causal_mask
from .models.transformer import BinaryToTextTransformer
from .tokenizer import load_tokenizer

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# =========================================
# Configuration
# =========================================

STEPS = 500

MAX_CIPHER_LENGTH = 4096

VOCAB_SIZE = 8000

D_MODEL = 256
NUM_HEADS = 8
D_FF = 1024
NUM_LAYERS = 2

LEARNING_RATE = 3e-4


# =========================================
# Dataset
# =========================================

dataset = BinaryToTextDataset(
    "data/brown_cipher.txt",
    "data/brown_plain.txt",
)

# Pick ONE example.
example = dataset[0]

cipher = example["cipher"].unsqueeze(0)

decoder_input = (
    example["decoder_input"]
    .unsqueeze(0)
)

target = (
    example["target"]
    .unsqueeze(0)
)


# =========================================
# Truncate cipher if necessary
# =========================================

cipher = cipher[:, :MAX_CIPHER_LENGTH]


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
# Training
# =========================================

model.train()

for step in range(1, STEPS + 1):

    causal_mask = create_causal_mask(
        decoder_input.size(1),
        decoder_input.device,
    )

    output = model(
        cipher,
        decoder_input,
        decoder_self_attention_mask=causal_mask,
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

    if step == 1 or step % 50 == 0:
        print(
            f"Step {step:3d} | "
            f"Loss: {loss.item():.6f}"
        )


# =========================================
# Generation
# =========================================

model.eval()

tokenizer = load_tokenizer()

BOS_ID = tokenizer.token_to_id("[BOS]")
EOS_ID = tokenizer.token_to_id("[EOS]")

generated = model.generate(
    cipher,
    bos_token_id=BOS_ID,
    eos_token_id=EOS_ID,
    max_length=300,
)


# =========================================
# Decode original
# =========================================

target_ids = target[0].tolist()

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
# Decode generated
# =========================================

generated_ids = generated[0].tolist()

if generated_ids[0] == BOS_ID:
    generated_ids = generated_ids[1:]

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