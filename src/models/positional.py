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