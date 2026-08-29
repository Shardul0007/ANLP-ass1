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
MAX_GENERATION_LENGTH = 300

MAX_EXAMPLES = 20

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
# Select COMPLETE validation examples
# =========================================

complete_examples = []

for index in range(
    len(validation_dataset)
):

    example = validation_dataset[index]

    cipher_length = (
        example["cipher"].size(0)
    )

    if cipher_length <= MAX_CIPHER_LENGTH:
        complete_examples.append(
            index
        )


print(
    "\nComplete validation examples:",
    len(complete_examples),
)

print(
    "Total validation examples:",
    len(validation_dataset),
)

print(
    "Coverage:",
    f"{100 * len(complete_examples) / len(validation_dataset):.2f}%"
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

total_exact_matches = 0

total_generated_tokens = 0
total_target_tokens = 0

generation_failures = 0


# =========================================
# Evaluate
# =========================================

num_examples = min(
    MAX_EXAMPLES,
    len(complete_examples),
)

print(
    "\nEvaluating",
    num_examples,
    "complete examples..."
)


for example_number in range(
    num_examples
):

    index = complete_examples[
        example_number
    ]

    example = validation_dataset[
        index
    ]

    print("\n" + "=" * 70)

    print(
        f"EXAMPLE {example_number + 1}"
        f"/{num_examples}"
    )

    print("=" * 70)


    # =====================================
    # Cipher
    # =====================================

    cipher = (
        example["cipher"]
        .unsqueeze(0)
        .to(device)
    )

    cipher_length = cipher.size(1)

    print(
        "\nCipher length:",
        cipher_length,
    )


    # =====================================
    # Target
    # =====================================

    target_cpu = example["target"]

    target_ids = (
        target_cpu
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


    # =====================================
    # Decoder input
    # =====================================

    decoder_input = (
        example["decoder_input"]
        .unsqueeze(0)
        .to(device)
    )

    target = (
        example["target"]
        .unsqueeze(0)
        .to(device)
    )


    # =====================================
    # Masks
    # =====================================

    # No cipher padding because this is
    # a single complete example.

    cipher_padding_mask = torch.zeros(
        (
            1,
            1,
            1,
            cipher.size(1),
        ),
        dtype=torch.bool,
        device=device,
    )

    decoder_padding_mask = (
        target == 0
    ).unsqueeze(1).unsqueeze(2)


    # =====================================
    # Teacher-forced evaluation
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


    # =====================================
    # Token accuracy
    # =====================================

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

    accuracy = (
        correct / token_count
        if token_count > 0
        else 0.0
    )

    total_correct += correct
    total_tokens += token_count


    # =====================================
    # Autoregressive generation
    # =====================================

    with torch.no_grad():

        generated = model.generate(
            cipher,
            bos_token_id=BOS_ID,
            eos_token_id=EOS_ID,
            max_length=MAX_GENERATION_LENGTH,
        )


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


    # Stop at EOS

    if EOS_ID in generated_ids:

        generated_ids = (
            generated_ids[
                :generated_ids.index(EOS_ID)
            ]
        )

    else:

        generation_failures += 1


    # =====================================
    # Exact match
    # =====================================

    exact_match = (
        generated_ids
        == target_ids
    )

    if exact_match:
        total_exact_matches += 1


    # =====================================
    # Length statistics
    # =====================================

    total_target_tokens += len(
        target_ids
    )

    total_generated_tokens += len(
        generated_ids
    )


    # =====================================
    # Decode
    # =====================================

    original_text = tokenizer.decode(
        target_ids
    )

    generated_text = tokenizer.decode(
        generated_ids
    )


    # =====================================
    # Print
    # =====================================

    print(
        "\nTeacher-forced accuracy:",
        f"{accuracy * 100:.2f}%"
    )

    print("\nORIGINAL:")
    print(original_text)

    print("\nGENERATED:")
    print(generated_text)

    print(
        "\nTarget tokens:",
        len(target_ids),
    )

    print(
        "Generated tokens:",
        len(generated_ids),
    )

    print(
        "Exact match:",
        exact_match,
    )


# =========================================
# Final summary
# =========================================

overall_accuracy = (
    total_correct / total_tokens
    if total_tokens > 0
    else 0.0
)

average_target_length = (
    total_target_tokens / num_examples
    if num_examples > 0
    else 0.0
)

average_generated_length = (
    total_generated_tokens / num_examples
    if num_examples > 0
    else 0.0
)

exact_match_rate = (
    total_exact_matches / num_examples
    if num_examples > 0
    else 0.0
)

generation_failure_rate = (
    generation_failures / num_examples
    if num_examples > 0
    else 0.0
)


print("\n\n" + "=" * 70)
print("C1 — COMPLETE-INPUT EVALUATION")
print("=" * 70)

print(
    "Examples evaluated:",
    num_examples,
)

print(
    "Teacher-forced token accuracy:",
    f"{overall_accuracy * 100:.2f}%"
)

print(
    "Average target length:",
    f"{average_target_length:.2f}"
)

print(
    "Average generated length:",
    f"{average_generated_length:.2f}"
)

print(
    "Exact matches:",
    f"{total_exact_matches}/{num_examples}"
)

print(
    "Exact match rate:",
    f"{exact_match_rate * 100:.2f}%"
)

print(
    "Generation hit max length:",
    f"{generation_failures}/{num_examples}"
)

print(
    "Generation failure rate:",
    f"{generation_failure_rate * 100:.2f}%"
)

print("=" * 70)
