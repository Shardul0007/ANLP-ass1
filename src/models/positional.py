import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model,
        max_sequence_length=4096,
    ):
        super().__init__()

        # Create position indices:
        # [0, 1, 2, ..., max_sequence_length - 1]
        positions = torch.arange(
            max_sequence_length,
            dtype=torch.float32,
        ).unsqueeze(1)

        # Dimension indices:
        # [0, 2, 4, ..., d_model - 2]
        dimension_indices = torch.arange(
            0,
            d_model,
            2,
            dtype=torch.float32,
        )

        # 1 / 10000^(2i / d_model)
        div_term = torch.exp(
            dimension_indices
            * (-math.log(10000.0) / d_model)
        )

        # Create positional encoding matrix.
        positional_encoding = torch.zeros(
            max_sequence_length,
            d_model,
        )

        # Even dimensions use sine.
        positional_encoding[:, 0::2] = torch.sin(
            positions * div_term
        )

        # Odd dimensions use cosine.
        positional_encoding[:, 1::2] = torch.cos(
            positions * div_term
        )

        # Add batch dimension:
        # [1, max_sequence_length, d_model]
        positional_encoding = positional_encoding.unsqueeze(0)

        # Register as a buffer because it is not trainable.
        self.register_buffer(
            "positional_encoding",
            positional_encoding,
        )

    def forward(self, x):
        """
        x:
            [batch, sequence_length, d_model]

        returns:
            [batch, sequence_length, d_model]
        """

        sequence_length = x.size(1)

        if sequence_length > self.positional_encoding.size(1):
            raise ValueError(
                "Sequence length exceeds maximum "
                "positional encoding length"
            )

        return (
            x
            + self.positional_encoding[
                :, :sequence_length, :
            ]
        )


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) as described in RoFormer (Su et al., 2021).
    Encodes relative position directly by rotating query and key vectors in the complex plane.
    """

    def __init__(self, dim, max_len=8192, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_len)

    def _build_cache(self, max_len):
        t = torch.arange(
            max_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len_cached = max_len

    def forward(self, x, seq_len=None):
        """
        Returns cos and sin tensors shaped to broadcast with x: [1, 1, seq_len, head_dim]
        """
        if seq_len is None:
            seq_len = x.size(-2)

        if seq_len > self.max_seq_len_cached:
            self._build_cache(seq_len)

        cos = self.cos_cached[:seq_len, :].to(x.dtype)
        sin = self.sin_cached[:seq_len, :].to(x.dtype)

        # Broadcast for [batch, heads, seq_len, head_dim]
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Applies Rotary Position Embedding to query and key tensors.
    q, k: [batch, heads, seq_len, head_dim]
    cos, sin: [1, 1, seq_len, head_dim]
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_single(x, cos, sin):
    """Applies RoPE to a single tensor."""
    return (x * cos) + (rotate_half(x) * sin)


if __name__ == "__main__":
    batch_size = 2
    sequence_length = 10
    d_model = 512

    x = torch.randn(
        batch_size,
        sequence_length,
        d_model,
    )

    positional_encoding = SinusoidalPositionalEncoding(
        d_model=d_model,
        max_sequence_length=100,
    )

    output = positional_encoding(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    print(
        "Stored positional encoding shape:",
        positional_encoding.positional_encoding.shape,
    )

    print("\nPosition 0:")
    print(
        positional_encoding.positional_encoding[
            0, 0, :10
        ]
    )

    print("\nPosition 1:")
    print(
        positional_encoding.positional_encoding[
            0, 1, :10
        ]
    )

    print("\nPosition 2:")
    print(
        positional_encoding.positional_encoding[
            0, 2, :10
        ]
    )