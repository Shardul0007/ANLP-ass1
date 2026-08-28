import math

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
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

        self.attention = ScaledDotProductAttention()

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
    def __init__(self):
        super().__init__()

    def forward(
        self,
        query,
        key,
        value,
        mask=None,
    ):
        """
        query: [batch, heads, query_length, head_dim]
        key:   [batch, heads, key_length, head_dim]
        value: [batch, heads, key_length, head_dim]

        Returns:
            output:
                [batch, heads, query_length, head_dim]

            attention_weights:
                [batch, heads, query_length, key_length]
        """

        head_dim = query.size(-1)

        # 1. Compute QK^T
        scores = torch.matmul(
            query,
            key.transpose(-2, -1),
        )

        # 2. Scale by sqrt(d_k)
        scores = scores / math.sqrt(head_dim)

        # 3. Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(
                mask,
                float("-inf"),
            )

        # 4. Convert scores to probabilities
        attention_weights = torch.softmax(
            scores,
            dim=-1,
        )

        # 5. Weighted sum of values
        output = torch.matmul(
            attention_weights,
            value,
        )

        return output, attention_weights


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