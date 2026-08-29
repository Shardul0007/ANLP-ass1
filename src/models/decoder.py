import torch.nn as nn

from .attention import MultiHeadAttention
from .ffn import FeedForwardNetwork
from .norm import LayerNorm


class DecoderLayer(nn.Module):
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
        self.norm3 = LayerNorm(d_model)

        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.cross_attention = MultiHeadAttention(
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
        encoder_output,
        self_attention_mask=None,
        cross_attention_mask=None,
    ):
        # =====================================
        # 1. Masked Self Attention
        # =====================================

        normalized_x = self.norm1(x)

        self_attention_output, self_attention_weights = (
            self.self_attention(
                normalized_x,
                normalized_x,
                normalized_x,
                mask=self_attention_mask,
            )
        )

        x = x + self.dropout(
            self_attention_output
        )

        # =====================================
        # 2. Cross Attention
        # =====================================

        normalized_x = self.norm2(x)

        cross_attention_output, cross_attention_weights = (
            self.cross_attention(
                normalized_x,
                encoder_output,
                encoder_output,
                mask=cross_attention_mask,
            )
        )

        x = x + self.dropout(
            cross_attention_output
        )

        # =====================================
        # 3. Feed Forward
        # =====================================

        normalized_x = self.norm3(x)

        ffn_output = self.ffn(
            normalized_x
        )

        x = x + self.dropout(
            ffn_output
        )

        return (
            x,
            self_attention_weights,
            cross_attention_weights,
        )