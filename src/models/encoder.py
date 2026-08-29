import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .ffn import FeedForwardNetwork
from .norm import LayerNorm


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout=0.1,
    ):
        super().__init__()

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.ffn = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        padding_mask=None,
    ):
        # =====================================
        # Pre-LN Self Attention
        # =====================================

        normalized_x = self.norm1(x)

        attention_output, attention_weights = (
            self.self_attention(
                normalized_x,
                normalized_x,
                normalized_x,
                mask=padding_mask,
            )
        )

        x = x + self.dropout(
            attention_output
        )

        # =====================================
        # Pre-LN Feed Forward
        # =====================================

        normalized_x = self.norm2(x)

        ffn_output = self.ffn(
            normalized_x
        )

        x = x + self.dropout(
            ffn_output
        )

        return x, attention_weights