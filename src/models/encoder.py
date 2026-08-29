import torch
import torch.nn as nn

from .attention import GroupedQueryAttention, MultiHeadAttention
from .ffn import FeedForwardNetwork
from .norm import LayerNorm, RMSNorm


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout=0.1,
        use_rope=False,
        norm_type="layernorm",
        attention_type="mha",
        num_kv_heads=None,
        max_len=8192,
    ):
        super().__init__()

        norm_cls = RMSNorm if norm_type.lower() == "rmsnorm" else LayerNorm
        self.norm1 = norm_cls(d_model)
        self.norm2 = norm_cls(d_model)

        if attention_type.lower() == "gqa":
            kv_heads = (
                num_kv_heads if num_kv_heads is not None else max(1, num_heads // 2)
            )
            self.self_attention = GroupedQueryAttention(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=kv_heads,
                dropout=dropout,
            )
        else:
            self.self_attention = MultiHeadAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                use_rope=use_rope,
                max_len=max_len,
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