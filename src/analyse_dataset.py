from .dataset import BinaryToTextDataset


dataset = BinaryToTextDataset(
    "data/brown_cipher.txt",
    "data/brown_plain.txt",
)

cipher_lengths = [
    len(sequence)
    for sequence in dataset.cipher_lines
]

text_lengths = []

for text in dataset.plain_lines:
    encoded = dataset.tokenizer.encode(text)
    text_lengths.append(len(encoded.ids))


def percentile(values, p):
    values = sorted(values)

    index = int(
        (len(values) - 1) * p
    )

    return values[index]


print("Number of examples:", len(dataset))

print("\nCipher lengths")
print("Min:", min(cipher_lengths))
print("Max:", max(cipher_lengths))
print("Mean:", sum(cipher_lengths) / len(cipher_lengths))
print("Median:", percentile(cipher_lengths, 0.50))
print("90th percentile:", percentile(cipher_lengths, 0.90))
print("95th percentile:", percentile(cipher_lengths, 0.95))
print("99th percentile:", percentile(cipher_lengths, 0.99))

print("\nPlaintext token lengths")
print("Min:", min(text_lengths))
print("Max:", max(text_lengths))
print("Mean:", sum(text_lengths) / len(text_lengths))
print("Median:", percentile(text_lengths, 0.50))
print("90th percentile:", percentile(text_lengths, 0.90))
print("95th percentile:", percentile(text_lengths, 0.95))
print("99th percentile:", percentile(text_lengths, 0.99))