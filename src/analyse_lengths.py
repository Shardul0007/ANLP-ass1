from .dataset import BinaryToTextDataset


dataset = BinaryToTextDataset(
    "data/brown_cipher.txt",
    "data/brown_plain.txt",
)

lengths = [
    len(cipher)
    for cipher in dataset.cipher_lines
]

total = len(lengths)

print(f"Total examples: {total}")

for limit in [512, 1024, 2048, 4096, 8192, 12288, 16384, 21360]:

    count = sum(
        length <= limit
        for length in lengths
    )

    percentage = 100 * count / total

    print(
        f"{limit:5d} bits: "
        f"{count:4d}/{total} "
        f"({percentage:6.2f}%)"
    )