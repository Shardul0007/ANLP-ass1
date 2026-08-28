import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()

        self.eps = eps

        # Learned scale and shift.
        self.gamma = nn.Parameter(
            torch.ones(d_model)
        )

        self.beta = nn.Parameter(
            torch.zeros(d_model)
        )

    def forward(self, x):
        """
        x:
            [..., d_model]
        """

        mean = x.mean(
            dim=-1,
            keepdim=True,
        )

        variance = (
            (x - mean) ** 2
        ).mean(
            dim=-1,
            keepdim=True,
        )

        normalized = (
            (x - mean)
            / torch.sqrt(variance + self.eps)
        )

        return (
            self.gamma * normalized
            + self.beta
        )


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()

        self.eps = eps

        # RMSNorm uses a learned scale.
        self.gamma = nn.Parameter(
            torch.ones(d_model)
        )

    def forward(self, x):
        """
        x:
            [..., d_model]
        """

        rms = torch.sqrt(
            (x ** 2).mean(
                dim=-1,
                keepdim=True,
            )
            + self.eps
        )

        normalized = x / rms

        return self.gamma * normalized


if __name__ == "__main__":
    batch_size = 2
    sequence_length = 10
    d_model = 512

    x = torch.randn(
        batch_size,
        sequence_length,
        d_model,
    )

    layer_norm = LayerNorm(d_model)
    rms_norm = RMSNorm(d_model)

    layer_norm_output = layer_norm(x)
    rms_norm_output = rms_norm(x)

    print("Input shape:", x.shape)

    print(
        "LayerNorm output shape:",
        layer_norm_output.shape,
    )

    print(
        "RMSNorm output shape:",
        rms_norm_output.shape,
    )

    # Check LayerNorm statistics.
    layer_mean = layer_norm_output.mean(
        dim=-1
    )

    layer_variance = layer_norm_output.var(
        dim=-1,
        unbiased=False,
    )

    print(
        "\nLayerNorm mean:",
        layer_mean[0, :5],
    )

    print(
        "LayerNorm variance:",
        layer_variance[0, :5],
    )

    # Check that parameters are learnable.
    print(
        "\nLayerNorm parameters:",
        sum(
            p.numel()
            for p in layer_norm.parameters()
        ),
    )

    print(
        "RMSNorm parameters:",
        sum(
            p.numel()
            for p in rms_norm.parameters()
        ),
    )