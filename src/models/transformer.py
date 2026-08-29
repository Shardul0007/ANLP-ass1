import math

import torch
import torch.nn as nn

from .decoder import DecoderLayer
from .encoder import EncoderLayer
from .positional import SinusoidalPositionalEncoding
from .masks import create_causal_mask
from .norm import LayerNorm

class Encoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        dropout=0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x,
        padding_mask=None,
    ):
        attention_weights = []

        for layer in self.layers:
            x, weights = layer(
                x,
                padding_mask=padding_mask,
            )

            attention_weights.append(weights)

        return x, attention_weights


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        dropout=0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x,
        encoder_output,
        self_attention_mask=None,
        cross_attention_mask=None,
    ):
        self_attention_weights = []
        cross_attention_weights = []

        for layer in self.layers:
            (
                x,
                self_weights,
                cross_weights,
            ) = layer(
                x,
                encoder_output,
                self_attention_mask,
                cross_attention_mask,
            )

            self_attention_weights.append(
                self_weights
            )

            cross_attention_weights.append(
                cross_weights
            )

        return (
            x,
            self_attention_weights,
            cross_attention_weights,
        )


class BinaryToTextTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        num_heads=8,
        d_ff=1024,
        num_layers=2,
        max_cipher_length=12000,
        max_text_length=1024,
        dropout=0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # =========================================
        # Embeddings
        # =========================================

        # Vocab: 0='0', 1='1', 2='[PAD]'
        self.binary_embedding = nn.Embedding(
            num_embeddings=3,
            embedding_dim=d_model,
            padding_idx=2,
        )

        self.text_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=0,
        )

        # =========================================
        # Positional encodings
        # =========================================

        self.encoder_position = (
            SinusoidalPositionalEncoding(
                d_model=d_model,
                max_sequence_length=max_cipher_length,
            )
        )

        self.decoder_position = (
            SinusoidalPositionalEncoding(
                d_model=d_model,
                max_sequence_length=max_text_length,
            )
        )

        # =========================================
        # Encoder
        # =========================================

        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.encoder_norm = LayerNorm(d_model)

        # =========================================
        # Decoder
        # =========================================

        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.decoder_norm = LayerNorm(d_model)

        # =========================================
        # Vocabulary projection
        # =========================================

        self.output_projection = nn.Linear(
            d_model,
            vocab_size,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        cipher,
        decoder_input,
        cipher_padding_mask=None,
        decoder_self_attention_mask=None,
        decoder_padding_mask=None,
    ):
        # =========================================
        # Encoder
        # =========================================

        encoder_x = self.binary_embedding(
            cipher
        )

        encoder_x = (
            encoder_x
            * math.sqrt(self.d_model)
        )

        encoder_x = self.encoder_position(
            encoder_x
        )

        encoder_x = self.dropout(
            encoder_x
        )

        encoder_output, encoder_attention = (
            self.encoder(
                encoder_x,
                padding_mask=cipher_padding_mask,
            )
        )
        encoder_output = self.encoder_norm(encoder_output)

        # =========================================
        # Decoder
        # =========================================

        decoder_x = self.text_embedding(
            decoder_input
        )

        decoder_x = (
            decoder_x
            * math.sqrt(self.d_model)
        )

        decoder_x = self.decoder_position(
            decoder_x
        )

        decoder_x = self.dropout(
            decoder_x
        )

        # Combine causal + padding masks.
        self_attention_mask = (
            decoder_self_attention_mask
        )

        if decoder_padding_mask is not None:
            if self_attention_mask is None:
                self_attention_mask = (
                    decoder_padding_mask
                )
            else:
                self_attention_mask = (
                    self_attention_mask
                    | decoder_padding_mask
                )

        decoder_output, decoder_self_attention, decoder_cross_attention = (
            self.decoder(
                decoder_x,
                encoder_output,
                self_attention_mask=self_attention_mask,
                cross_attention_mask=cipher_padding_mask,
            )
        )
        decoder_output = self.decoder_norm(decoder_output)

        # =========================================
        # Vocabulary prediction
        # =========================================

        logits = self.output_projection(
            decoder_output
        )

        return {
            "logits": logits,
            "encoder_attention": encoder_attention,
            "decoder_self_attention": decoder_self_attention,
            "decoder_cross_attention": decoder_cross_attention,
        }
    @torch.no_grad()
    def generate(
        self,
        cipher,
        bos_token_id,
        eos_token_id,
        max_length=128,
        cipher_padding_mask=None,
    ):
        """
        Autoregressively generate plaintext token IDs using greedy decoding.

        cipher:
            [batch, cipher_length]

        returns:
            [batch, generated_length]
        """

        self.eval()

        # Format mask if needed
        if cipher_padding_mask is not None and cipher_padding_mask.dim() == 2:
            cipher_padding_mask = (
                cipher_padding_mask.unsqueeze(1).unsqueeze(2)
            )

        # =========================================
        # Encoder
        # =========================================

        encoder_x = self.binary_embedding(cipher)

        encoder_x = (
            encoder_x
            * math.sqrt(self.d_model)
        )

        encoder_x = self.encoder_position(
            encoder_x
        )

        encoder_output, _ = self.encoder(
            encoder_x,
            padding_mask=cipher_padding_mask,
        )
        encoder_output = self.encoder_norm(encoder_output)

        # =========================================
        # Start with BOS
        # =========================================

        generated = torch.full(
            (cipher.size(0), 1),
            bos_token_id,
            dtype=torch.long,
            device=cipher.device,
        )

        finished = torch.zeros(
            cipher.size(0),
            dtype=torch.bool,
            device=cipher.device,
        )

        # =========================================
        # Autoregressive generation
        # =========================================

        for _ in range(max_length):

            decoder_x = self.text_embedding(
                generated
            )

            decoder_x = (
                decoder_x
                * math.sqrt(self.d_model)
            )

            decoder_x = self.decoder_position(
                decoder_x
            )

            # Causal mask
            causal_mask = create_causal_mask(
                generated.size(1),
                generated.device,
            )

            decoder_output, _, _ = self.decoder(
                decoder_x,
                encoder_output,
                self_attention_mask=causal_mask,
                cross_attention_mask=cipher_padding_mask,
            )
            decoder_output = self.decoder_norm(decoder_output)

            # Only need prediction for the newest token.
            last_hidden = decoder_output[:, -1, :]

            logits = self.output_projection(
                last_hidden
            )

            next_token = logits.argmax(
                dim=-1
            )

            generated = torch.cat(
                [
                    generated,
                    next_token.unsqueeze(1),
                ],
                dim=1,
            )

            finished |= (
                next_token == eos_token_id
            )

            if finished.all():
                break

        return generated
if __name__ == "__main__":
    from ..tokenizer import load_tokenizer

    tokenizer = load_tokenizer()

    model = BinaryToTextTransformer(
        vocab_size=8000,
        d_model=256,
        num_heads=8,
        d_ff=1024,
        num_layers=2,
        max_cipher_length=512,
        max_text_length=1024,
    )

    cipher = torch.randint(
        0,
        2,
        (1, 128),
    )

    generated = model.generate(
        cipher,
        bos_token_id=tokenizer.token_to_id("[BOS]"),
        eos_token_id=tokenizer.token_to_id("[EOS]"),
        max_length=20,
    )

    print("Generated IDs:")
    print(generated)

    print("\nGenerated tokens:")
    print(
        tokenizer.decode(
            generated[0].tolist()
        )
    )