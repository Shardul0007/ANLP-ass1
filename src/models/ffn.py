import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
    ):
        super().__init__()

        self.linear1 = nn.Linear(
            d_model,
            d_ff,
        )

        self.activation = nn.GELU()

        self.linear2 = nn.Linear(
            d_ff,
            d_model,
        )

    def forward(self, x):
        """
        x:
            [batch, sequence_length, d_model]

        returns:
            [batch, sequence_length, d_model]
        """

        x = self.linear1(x)

        x = self.activation(x)

        x = self.linear2(x)

        return x


if __name__ == "__main__":
    batch_size = 2
    sequence_length = 10
    d_model = 512
    d_ff = 2048

    x = torch.randn(
        batch_size,
        sequence_length,
        d_model,
    )

    ffn = FeedForwardNetwork(
        d_model=d_model,
        d_ff=d_ff,
    )

    output = ffn(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    print(
        "Linear 1:",
        ffn.linear1.weight.shape,
    )

    print(
        "Linear 2:",
        ffn.linear2.weight.shape,
    )