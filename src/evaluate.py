import torch
from torch.utils.data import random_split

from .dataset import BinaryToTextDataset
from .models.masks import create_causal_mask
from .models.transformer import BinaryToTextTransformer
from .tokenizer import load_tokenizer


# =========================================
# Configuration
# =========================================

CHECKPOINT = "checkpoints/c1_best.pt"

MAX_CIPHER_LENGTH = 4096
MAX_GENERATION_LENGTH = 700

NUM_EXAMPLES = 20

VOCAB_SIZE = 8000

D_MODEL = 256
NUM_HEADS = 8
D_FF = 1024
NUM_LAYERS = 2


# =========================================
# Device
# =========================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


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
).to(device)


# =========================================
# Load checkpoint
# =========================================

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
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
# Aggregate metrics
# =========================================

total_correct = 0
total_tokens = 0

total_generated_tokens = 0
total_target_tokens = 0

total_exact_matches = 0
total_early_eos = 0

# =========================================
# Evaluate validation examples
# =========================================

num_examples = min(
    NUM_EXAMPLES,
    len(validation_dataset),
)

for index in range(num_examples):

    print("\n" + "=" * 70)
    print(
        f"EXAMPLE {index + 1}/{num_examples}"
    )
    print("=" * 70)

    example = validation_dataset[index]

    # =====================================
    # Cipher
    # =====================================

    cipher = (
        example["cipher"]
        .unsqueeze(0)
        .to(device)
    )

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


    # =====================================
    # Target / decoder input
    # =====================================

    target = (
        example["target"]
        .unsqueeze(0)
        .to(device)
    )

    decoder_input = (
        example["decoder_input"]
        .unsqueeze(0)
        .to(device)
    )

    # =====================================
    # Padding masks
    # =====================================

    cipher_padding_mask = (
        example["cipher_padding_mask"]
        .unsqueeze(0)
        .unsqueeze(1)
        .unsqueeze(2)
        .to(device)
    )

    decoder_padding_mask = (
        example["target_padding_mask"]
        .unsqueeze(0)
        .unsqueeze(1)
        .unsqueeze(2)
        .to(device)
    )

    # =====================================
    # Original plaintext
    # =====================================

    target_ids = (
        example["target"]
        .tolist()
    )

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


    # =====================================
    # 1. Teacher-forced evaluation
    # =====================================

    with torch.no_grad():

        causal_mask = create_causal_mask(
            decoder_input.size(1),
            device,
        )

        output = model(
            cipher,
            decoder_input,
            cipher_padding_mask=(
                cipher_padding_mask
            ),
            decoder_self_attention_mask=(
                causal_mask
            ),
            decoder_padding_mask=(
                decoder_padding_mask
            ),
        )

        logits = output["logits"]

        predictions = logits.argmax(
            dim=-1
        )

    # Ignore padding tokens
    valid_positions = (
        target != 0
    )

    correct = (
        (
            predictions[valid_positions]
            == target[valid_positions]
        )
        .sum()
        .item()
    )

    token_count = (
        valid_positions
        .sum()
        .item()
    )

    teacher_forced_accuracy = (
        correct / token_count
        if token_count > 0
        else 0.0
    )

    total_correct += correct
    total_tokens += token_count


    # =====================================
    # 2. Autoregressive generation
    # =====================================

    with torch.no_grad():

        generated = model.generate(
            cipher,
            bos_token_id=BOS_ID,
            eos_token_id=EOS_ID,
            max_length=MAX_GENERATION_LENGTH,
        )


    # =====================================
    # Decode prediction
    # =====================================

    generated_ids = (
        generated[0]
        .detach()
        .cpu()
        .tolist()
    )

    # Remove BOS
    if (
        generated_ids
        and generated_ids[0] == BOS_ID
    ):
        generated_ids = (
            generated_ids[1:]
        )

    # Check whether EOS was generated
    generated_eos = (
        EOS_ID in generated_ids
    )

    if generated_eos:

        generated_ids = (
            generated_ids[
                :generated_ids.index(EOS_ID)
            ]
        )

    else:

        total_early_eos += 1

    generated_text = tokenizer.decode(
        generated_ids
    )


    # =====================================
    # Autoregressive statistics
    # =====================================

    generated_token_count = len(
        generated_ids
    )

    target_token_count = len(
        target_ids
    )

    total_generated_tokens += (
        generated_token_count
    )

    total_target_tokens += (
        target_token_count
    )

    # Exact sequence match
    if generated_ids == target_ids:
        total_exact_matches += 1


    # =====================================
    # Print example results
    # =====================================

    print(
        "\nTeacher-forced token accuracy:",
        f"{teacher_forced_accuracy * 100:.2f}%"
    )

    print("\nORIGINAL:")
    print(original_text)

    print("\nGENERATED:")
    print(generated_text)

    print(
        "\nOriginal token count:",
        target_token_count,
    )

    print(
        "Generated token count:",
        generated_token_count,
    )

    print(
        "EOS generated:",
        generated_eos,
    )


# =========================================
# Final aggregate results
# =========================================

overall_accuracy = (
    total_correct / total_tokens
    if total_tokens > 0
    else 0.0
)

average_generated_length = (
    total_generated_tokens / num_examples
)

average_target_length = (
    total_target_tokens / num_examples
)

exact_match_rate = (
    total_exact_matches / num_examples
)

print("\n\n" + "=" * 70)
print("C1 EVALUATION SUMMARY")
print("=" * 70)

print(
    f"Examples evaluated: "
    f"{num_examples}"
)

print(
    f"Teacher-forced token accuracy: "
    f"{overall_accuracy * 100:.2f}%"
)

print(
    f"Average target length: "
    f"{average_target_length:.2f} tokens"
)

print(
    f"Average generated length: "
    f"{average_generated_length:.2f} tokens"
)

print(
    f"Exact sequence matches: "
    f"{total_exact_matches}/{num_examples}"
)

print(
    f"Exact sequence match rate: "
    f"{exact_match_rate * 100:.2f}%"
)

print(
    f"Generation hit max length: "
    f"{total_early_eos}/{num_examples}"
)

print("=" * 70)