from __future__ import annotations
import time
from src.tokenizer import BPETokenizer

print("1. Starting debug test...")
sample_ciphers = [
    "01101000|01100101|01101100|01101100|01101111|",
    "01110111|01101111|01110010|01101100|01100100|",
]
sample_plains = ["hello world this is a test", "another sentence for testing bpe"]

t0 = time.time()
print("2. Training cipher tokenizer...")
tok_c = BPETokenizer.train(sample_ciphers, vocab_size=50, is_cipher=True)
print(f"Cipher tokenizer trained in {time.time() - t0:.4f}s, vocab size: {tok_c.get_vocab_size()}")

t0 = time.time()
print("3. Training plain tokenizer...")
tok_p = BPETokenizer.train(sample_plains, vocab_size=50, is_cipher=False)
print(f"Plain tokenizer trained in {time.time() - t0:.4f}s, vocab size: {tok_p.get_vocab_size()}")

t0 = time.time()
enc_c = tok_c.encode(sample_ciphers[0])
print(f"Cipher encoded in {time.time() - t0:.4f}s: {enc_c.ids}")

t0 = time.time()
enc_p = tok_p.encode(sample_plains[0])
print(f"Plain encoded in {time.time() - t0:.4f}s: {enc_p.ids}")
print("Debug test passed successfully!")
