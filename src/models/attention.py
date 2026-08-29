import math

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Learned projections for Q, K and V.
        self.q_projection = nn.Linear(
            d_model,
            d_model,
        )

        self.k_projection = nn.Linear(
            d_model,
            d_model,
        )

        self.v_projection = nn.Linear(
            d_model,
            d_model,
        )

        # Final projection after concatenating heads.
        self.output_projection = nn.Linear(
            d_model,
            d_model,
        )

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def split_heads(self, x):
        """
        x:
            [batch, sequence_length, d_model]

        returns:
            [batch, num_heads, sequence_length, head_dim]
        """

        batch_size, sequence_length, _ = x.shape

        x = x.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        x = x.transpose(1, 2)

        return x

    def combine_heads(self, x):
        """
        x:
            [batch, num_heads, sequence_length, head_dim]

        returns:
            [batch, sequence_length, d_model]
        """

        batch_size, num_heads, sequence_length, head_dim = x.shape

        x = x.transpose(1, 2)

        x = x.contiguous().view(
            batch_size,
            sequence_length,
            self.d_model,
        )

        return x

    def forward(
        self,
        query,
        key,
        value,
        mask=None,
        need_weights=False,
    ):
        """
        query:
            [batch, query_length, d_model]

        key:
            [batch, key_length, d_model]

        value:
            [batch, key_length, d_model]
        """

        # 1. Project inputs into Q, K and V.
        q = self.q_projection(query)
        k = self.k_projection(key)
        v = self.v_projection(value)

        # 2. Split into multiple heads.
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # 3. Perform scaled dot-product attention.
        attention_output, attention_weights = self.attention(
            q,
            k,
            v,
            mask,
            need_weights=need_weights,
        )

        # 4. Combine heads.
        attention_output = self.combine_heads(
            attention_output
        )

        # 5. Final learned projection.
        output = self.output_projection(
            attention_output
        )

        return output, attention_weights

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.0, chunk_size=1024):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.chunk_size = chunk_size

    def _compute_chunk(self, query, key, value, mask=None, need_weights=False):
        head_dim = query.size(-1)

        scores = torch.matmul(
            query,
            key.transpose(-2, -1),
        )
        scores = scores / math.sqrt(head_dim)

        if mask is not None:
            fill_value = -1e4 if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(
                mask,
                fill_value,
            )

        attention_weights = torch.softmax(
            scores,
            dim=-1,
        )
        del scores

        attention_weights_dropped = self.dropout(attention_weights)

        output = torch.matmul(
            attention_weights_dropped,
            value,
        )

        if need_weights:
            return output, attention_weights
        return output, None

    def forward(
        self,
        query,
        key,
        value,
        mask=None,
        need_weights=False,
    ):
        """
        query: [batch, heads, query_length, head_dim]
        key:   [batch, heads, key_length, head_dim]
        value: [batch, heads, key_length, head_dim]
        """
        q_len = query.size(-2)

        # Fast path for standard length or when chunking is not beneficial
        if q_len <= self.chunk_size:
            return self._compute_chunk(query, key, value, mask, need_weights=need_weights)

        # Memory-efficient chunked computation across query dimension:
        # Splits query into blocks of chunk_size to keep peak attention memory small
        outputs = []
        weights = [] if need_weights else None

        for i in range(0, q_len, self.chunk_size):
            q_chunk = query[:, :, i : i + self.chunk_size, :]
            m_chunk = (
                mask[:, :, i : i + self.chunk_size, :]
                if mask is not None and mask.size(-2) > 1
                else mask
            )
            out_c, w_c = self._compute_chunk(
                q_chunk, key, value, m_chunk, need_weights=need_weights
            )
            outputs.append(out_c)
            if need_weights and w_c is not None:
                weights.append(w_c)

        output = torch.cat(outputs, dim=-2)
        total_weights = torch.cat(weights, dim=-2) if need_weights else None

        return output, total_weights


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, dropout=0.0, chunk_size=1024):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.head_dim = d_model // num_heads

        self.q_projection = nn.Linear(d_model, d_model)
        self.k_projection = nn.Linear(d_model, self.num_kv_heads * self.head_dim)
        self.v_projection = nn.Linear(d_model, self.num_kv_heads * self.head_dim)
        self.output_projection = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout, chunk_size=chunk_size)

    def split_heads(self, x, num_heads):
        batch_size, sequence_length, _ = x.shape
        x = x.view(batch_size, sequence_length, num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, query, key, value, mask=None, need_weights=False):
        q = self.q_projection(query)
        k = self.k_projection(key)
        v = self.v_projection(value)

        q = self.split_heads(q, self.num_heads)
        k = self.split_heads(k, self.num_kv_heads)
        v = self.split_heads(v, self.num_kv_heads)

        # Repeat KV heads to match query heads
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        attention_output, attention_weights = self.attention(
            q, k, v, mask=mask, need_weights=need_weights
        )

        # Combine heads
        batch_size, _, sequence_length, _ = attention_output.shape
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.d_model)
        )

        return self.output_projection(attention_output), attention_weights


if __name__ == "__main__":
    batch_size = 2
    sequence_length = 10
    d_model = 512
    num_heads = 8

    x = torch.randn(
        batch_size,
        sequence_length,
        d_model,
    )

    mha = MultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
    )

    output, weights = mha(
        x,
        x,
        x,
    )

    print("Input shape:", x.shape)

    print(
        "Attention weights shape:",
        weights.shape,
    )

    print("Output shape:", output.shape)

    print("\nExpected:")
    print(
        "Weights:",
        f"[{batch_size}, {num_heads}, "
        f"{sequence_length}, {sequence_length}]",
    )

    print(
        "Output:",
        f"[{batch_size}, {sequence_length}, {d_model}]",
    )