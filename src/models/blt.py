import math
from typing import Optional, List, Tuple

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .norm import LayerNorm
from .positional import SinusoidalPositionalEncoding
from .masks import create_causal_mask
from .encoder import EncoderLayer
from .decoder import DecoderLayer


NUM_BYTE_CLASSES = 259  # 0: [PAD], 1: [BOS], 2: [EOS], 3..258: byte values 0..255
PAD_BYTE_ID = 0
BOS_BYTE_ID = 1
EOS_BYTE_ID = 2


def string_to_byte_ids(text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
    """Converts a string to byte IDs with offset + 3 for special tokens."""
    raw = list(text.encode("utf-8"))
    ids = [b + 3 for b in raw]
    if add_bos:
        ids = [BOS_BYTE_ID] + ids
    if add_eos:
        ids = ids + [EOS_BYTE_ID]
    return ids


def byte_ids_to_string(ids: List[int]) -> str:
    """Decodes a list of byte IDs back to a UTF-8 string."""
    raw = []
    for token_id in ids:
        if token_id == EOS_BYTE_ID:
            break
        if token_id >= 3:
            raw.append(token_id - 3)
    return bytes(raw).decode("utf-8", errors="replace")


class LocalCipherEncoder(nn.Module):
    """
    Local Encoder for Ciphertext:
    Groups raw cipher bits into patches of size (patch_size_bytes * 8) bits
    and maps each bit-patch to a latent representation of dimension d_model.
    """

    def __init__(self, patch_size_bytes: int = 4, d_model: int = 256):
        super().__init__()
        self.patch_size_bits = patch_size_bytes * 8
        self.d_model = d_model

        # Bit embedding: 0, 1, and 2 for [PAD]
        self.bit_emb = nn.Embedding(3, 16, padding_idx=2)
        self.patch_proj = nn.Sequential(
            nn.Linear(self.patch_size_bits * 16, d_model),
            LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, cipher_bits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        cipher_bits: [batch, cipher_len_bits]
        returns:
            patch_latents: [batch, num_patches, d_model]
            patch_padding_mask: [batch, 1, 1, num_patches]
        """
        batch_size, bit_len = cipher_bits.shape

        # Pad bit_len to a multiple of patch_size_bits if needed
        rem = bit_len % self.patch_size_bits
        if rem != 0:
            pad_amount = self.patch_size_bits - rem
            pad_tensor = torch.full(
                (batch_size, pad_amount), 2, dtype=torch.long, device=cipher_bits.device
            )
            cipher_bits = torch.cat([cipher_bits, pad_tensor], dim=1)
            bit_len = cipher_bits.shape[1]

        num_patches = bit_len // self.patch_size_bits

        # Embed each bit
        x = self.bit_emb(cipher_bits)  # [batch, bit_len, 16]
        # Reshape to patches
        x = x.view(batch_size, num_patches, self.patch_size_bits * 16)
        patch_latents = self.patch_proj(x)  # [batch, num_patches, d_model]

        # Patch padding mask: True where all bits in patch are PAD (value 2)
        bit_patches = cipher_bits.view(batch_size, num_patches, self.patch_size_bits)
        is_pad_patch = (bit_patches == 2).all(dim=-1)  # [batch, num_patches]
        patch_padding_mask = is_pad_patch.unsqueeze(1).unsqueeze(2)

        return patch_latents, patch_padding_mask


class LocalByteEncoder(nn.Module):
    """
    Local Encoder for Plaintext Bytes:
    Embeds raw bytes and groups them into patches of size patch_size_bytes,
    projecting each patch into a latent representation of dimension d_model.
    """

    def __init__(
        self,
        patch_size_bytes: int = 4,
        d_byte: int = 128,
        d_model: int = 256,
        num_bytes: int = NUM_BYTE_CLASSES,
    ):
        super().__init__()
        self.patch_size_bytes = patch_size_bytes
        self.d_model = d_model
        self.byte_embedding = nn.Embedding(num_bytes, d_byte, padding_idx=PAD_BYTE_ID)
        self.patch_proj = nn.Sequential(
            nn.Linear(patch_size_bytes * d_byte, d_model),
            LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, byte_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        byte_seq: [batch, byte_len]
        returns:
            patch_latents: [batch, num_patches, d_model]
            patch_padding_mask: [batch, 1, 1, num_patches]
        """
        batch_size, byte_len = byte_seq.shape

        # Pad byte_len to a multiple of patch_size_bytes if needed
        rem = byte_len % self.patch_size_bytes
        if rem != 0:
            pad_amount = self.patch_size_bytes - rem
            pad_tensor = torch.full(
                (batch_size, pad_amount), PAD_BYTE_ID, dtype=torch.long, device=byte_seq.device
            )
            byte_seq = torch.cat([byte_seq, pad_tensor], dim=1)
            byte_len = byte_seq.shape[1]

        num_patches = byte_len // self.patch_size_bytes

        # Embed each byte
        x = self.byte_embedding(byte_seq)  # [batch, byte_len, d_byte]
        # Reshape to patches
        x = x.view(batch_size, num_patches, self.patch_size_bytes * x.size(-1))
        patch_latents = self.patch_proj(x)  # [batch, num_patches, d_model]

        # Patch padding mask: True where all bytes in patch are PAD (0)
        byte_patches = byte_seq.view(batch_size, num_patches, self.patch_size_bytes)
        is_pad_patch = (byte_patches == PAD_BYTE_ID).all(dim=-1)
        patch_padding_mask = is_pad_patch.unsqueeze(1).unsqueeze(2)

        return patch_latents, patch_padding_mask


class LocalByteDecoder(nn.Module):
    """
    Local Decoder:
    Expands each latent patch vector from the global transformer back into
    patch_size_bytes individual byte predictions over 259 byte classes.
    """

    def __init__(
        self,
        patch_size_bytes: int = 4,
        d_model: int = 256,
        d_byte: int = 128,
        num_bytes: int = NUM_BYTE_CLASSES,
    ):
        super().__init__()
        self.patch_size_bytes = patch_size_bytes
        self.num_bytes = num_bytes
        self.unpatch = nn.Sequential(
            nn.Linear(d_model, d_model),
            LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, patch_size_bytes * d_byte),
        )
        self.byte_head = nn.Linear(d_byte, num_bytes)

    def forward(self, patch_latents: torch.Tensor) -> torch.Tensor:
        """
        patch_latents: [batch, num_patches, d_model]
        returns:
            byte_logits: [batch, num_patches * patch_size_bytes, num_bytes]
        """
        batch_size, num_patches, _ = patch_latents.shape
        x = self.unpatch(patch_latents)  # [batch, num_patches, patch_size * d_byte]
        x = x.view(batch_size, num_patches * self.patch_size_bytes, -1)
        return self.byte_head(x)  # [batch, total_bytes, num_bytes]


class ByteLatentTransformer(nn.Module):
    """
    Configuration 5 (C5): Byte Latent Transformer (Token-Free BLT).

    - Local Cipher Encoder: Groups cipher bits into 32-bit patches -> latents.
    - Local Plaintext Encoder: Groups bytes into 4-byte patches -> latents.
    - Global Transformer: 2 Encoder layers, 2 Decoder layers with MHA, Pre-LN LayerNorm,
      and Sinusoidal Positional Encodings (matching Table 1 C1 specs).
    - Local Decoder: Projects output latents back to raw byte distributions.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        d_ff: int = 1024,
        num_layers: int = 2,
        patch_size_bytes: int = 4,
        d_byte: int = 128,
        max_patches: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_size_bytes = patch_size_bytes

        # Local Encoders & Decoders
        self.local_cipher_encoder = LocalCipherEncoder(
            patch_size_bytes=patch_size_bytes, d_model=d_model
        )
        self.local_byte_encoder = LocalByteEncoder(
            patch_size_bytes=patch_size_bytes,
            d_byte=d_byte,
            d_model=d_model,
        )
        self.local_byte_decoder = LocalByteDecoder(
            patch_size_bytes=patch_size_bytes,
            d_model=d_model,
            d_byte=d_byte,
        )

        # Positional Encodings in Latent Space (Sinusoidal, as required by Table 1)
        self.encoder_pos = SinusoidalPositionalEncoding(
            d_model=d_model, max_sequence_length=max_patches
        )
        self.decoder_pos = SinusoidalPositionalEncoding(
            d_model=d_model, max_sequence_length=max_patches
        )

        # Global Transformer Encoder & Decoder (MHA, LayerNorm)
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    use_rope=False,
                    norm_type="layernorm",
                    attention_type="mha",
                )
                for _ in range(num_layers)
            ]
        )
        self.encoder_norm = LayerNorm(d_model)

        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    use_rope=False,
                    norm_type="layernorm",
                    attention_type="mha",
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def encode(
        self, cipher_bits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs local cipher patching and global transformer encoder."""
        cipher_latents, cipher_mask = self.local_cipher_encoder(cipher_bits)
        x = cipher_latents * math.sqrt(self.d_model)
        x = self.encoder_pos(x)
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x, _ = layer(x, padding_mask=cipher_mask)
        x = self.encoder_norm(x)
        return x, cipher_mask

    def decode(
        self,
        decoder_bytes: torch.Tensor,
        encoder_output: torch.Tensor,
        cipher_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Runs local byte patching, global transformer decoder, and local byte decoder."""
        patch_latents, _ = self.local_byte_encoder(decoder_bytes)
        num_patches = patch_latents.size(1)

        x = patch_latents * math.sqrt(self.d_model)
        x = self.decoder_pos(x)
        x = self.dropout(x)

        # Causal mask over latent patches
        causal_mask = create_causal_mask(num_patches, x.device)

        for layer in self.decoder_layers:
            x, _, _ = layer(
                x,
                encoder_output,
                self_attention_mask=causal_mask,
                cross_attention_mask=cipher_mask,
            )
        x = self.decoder_norm(x)

        # Local Decoder: expand patches back to raw byte distributions
        byte_logits = self.local_byte_decoder(x)
        return byte_logits

    def forward(
        self, cipher: torch.Tensor, decoder_input: torch.Tensor, **kwargs
    ) -> dict:
        """
        cipher: [batch, cipher_bits_len]
        decoder_input: [batch, byte_seq_len]
        returns:
            dict with 'logits': [batch, num_decoded_bytes, NUM_BYTE_CLASSES]
        """
        enc_out, cipher_mask = self.encode(cipher)
        logits = self.decode(decoder_input, enc_out, cipher_mask)
        return {"logits": logits}

    @torch.no_grad()
    def generate(
        self,
        cipher: torch.Tensor,
        max_length_bytes: int = 400,
        bos_byte_id: int = BOS_BYTE_ID,
        eos_byte_id: int = EOS_BYTE_ID,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> torch.Tensor:
        """
        Autoregressively generates text byte-by-byte (patch-by-patch).
        cipher: [batch, cipher_bits_len]
        returns:
            generated_bytes: [batch, generated_byte_len]
        """
        self.eval()
        batch_size = cipher.size(0)
        device = cipher.device

        enc_out, cipher_mask = self.encode(cipher)

        # Start with [BOS] padded to one patch
        init_bytes = [bos_byte_id] + [PAD_BYTE_ID] * (self.patch_size_bytes - 1)
        generated = torch.tensor(
            [init_bytes] * batch_size, dtype=torch.long, device=device
        )

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        max_patches = max_length_bytes // self.patch_size_bytes

        for _ in range(max_patches):
            logits = self.decode(generated, enc_out, cipher_mask)
            last_patch_logits = logits[:, -self.patch_size_bytes :, :]  # [batch, patch_size, 259]

            if repetition_penalty > 1.0:
                for b in range(batch_size):
                    for prev_token in set(generated[b].tolist()):
                        if last_patch_logits[b, :, prev_token].mean() < 0:
                            last_patch_logits[b, :, prev_token] *= repetition_penalty
                        else:
                            last_patch_logits[b, :, prev_token] /= repetition_penalty

            new_bytes = last_patch_logits.argmax(dim=-1)

            for b in range(batch_size):
                if eos_byte_id in new_bytes[b].tolist():
                    finished[b] = True

            generated = torch.cat([generated, new_bytes], dim=1)
            if finished.all():
                break

        return generated
