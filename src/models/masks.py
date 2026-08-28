import torch


def create_causal_mask(sequence_length, device=None):
    """
    Create a causal attention mask.

    False = attention is allowed
    True  = attention is blocked

    Shape:
        [1, 1, sequence_length, sequence_length]
    """

    mask = torch.triu(
        torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=device,
        ),
        diagonal=1,
    )

    return mask.unsqueeze(0).unsqueeze(0)


def create_padding_mask(token_ids, padding_id):
    """
    token_ids:
        [batch, sequence_length]

    Returns:
        [batch, 1, 1, sequence_length]

    True  = padding
    False = real token
    """

    mask = token_ids == padding_id

    return mask.unsqueeze(1).unsqueeze(2)


if __name__ == "__main__":
    sequence_length = 5

    causal_mask = create_causal_mask(
        sequence_length
    )

    print("Causal mask:")
    print(causal_mask[0, 0])

    tokens = torch.tensor([
        [10, 20, 30, 0, 0],
        [40, 50, 60, 70, 0],
    ])

    padding_mask = create_padding_mask(
        tokens,
        padding_id=0,
    )

    print("\nPadding mask:")
    print(padding_mask[:, 0, 0])