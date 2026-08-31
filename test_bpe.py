from __future__ import annotations
from src.tokenizer import BPETokenizer
from src.dataset import build_dataloaders

print("Testing from-scratch BPETokenizer on sample...")
sample_ciphers = [
    "01101000|01100101|01101100|01101100|01101111|",
    "01110111|01101111|01110010|01101100|01100100|",
]
sample_plains = ["hello", "world"]

tok_c = BPETokenizer.train(sample_ciphers, vocab_size=30, is_cipher=True)
enc_c = tok_c.encode(sample_ciphers[0])
print("Cipher vocab size:", tok_c.get_vocab_size())
print("Cipher encoded IDs:", enc_c.ids)
print("Cipher decoded:", tok_c.decode(enc_c.ids))

tok_p = BPETokenizer.train(sample_plains, vocab_size=30, is_cipher=False)
enc_p = tok_p.encode("hello world")
print("Plain vocab size:", tok_p.get_vocab_size())
print("Plain encoded IDs:", enc_p.ids)
print("Plain decoded:", tok_p.decode(enc_p.ids))

print("\nTokenizer test passed successfully!")
