import unittest

import torch

from joeynmt.decoders import ConvDecoder
from joeynmt.encoders import ConvEncoder


class TestConvDecoder(unittest.TestCase):

    def setUp(self):
        self.emb_size = 8
        self.hidden_size = 16
        self.num_layers = 3
        self.kernel_width = 3
        self.vocab_size = 20
        self.max_position = 64
        self.seed = 42
        torch.manual_seed(self.seed)

    def _encoder(self, **overrides):
        cfg = {
            "hidden_size": self.hidden_size,
            "emb_size": self.emb_size,
            "num_layers": 2,
            "kernel_width": self.kernel_width,
            "dropout": 0.0,
            "emb_dropout": 0.0,
            "max_position": self.max_position,
        }
        cfg.update(overrides)
        torch.manual_seed(self.seed)
        encoder = ConvEncoder(**cfg)
        encoder.eval()
        return encoder

    def _decoder(self, **overrides):
        cfg = {
            "hidden_size": self.hidden_size,
            "emb_size": self.emb_size,
            "num_layers": self.num_layers,
            "kernel_width": self.kernel_width,
            "dropout": 0.0,
            "emb_dropout": 0.0,
            "vocab_size": self.vocab_size,
            "max_position": self.max_position,
        }
        cfg.update(overrides)
        torch.manual_seed(self.seed)
        decoder = ConvDecoder(**cfg)
        decoder.eval()
        return decoder

    def _encode(self, encoder, src_lengths, src_len=None):
        """Run the conv encoder and return (encoder_output, src_mask)."""
        batch_size = len(src_lengths)
        src_len = src_len if src_len is not None else max(src_lengths)
        src_embed = torch.rand(batch_size, src_len, self.emb_size)
        src_mask = torch.zeros(batch_size, 1, src_len, dtype=torch.bool)
        for i, length in enumerate(src_lengths):
            src_mask[i, 0, :length] = True
        encoder_output, _ = encoder(src_embed, torch.tensor(src_lengths), src_mask)
        return encoder_output, src_mask

    def test_conv_decoder_freeze(self):
        decoder = self._decoder(freeze=True)
        for _, p in decoder.named_parameters():
            self.assertFalse(p.requires_grad)

    def test_conv_decoder_output_shape(self):
        """logits (B, T, V), states (B, T, H), attention (B, T, S)."""
        batch_size, src_len = 3, 6
        encoder = self._encoder()
        encoder_output, src_mask = self._encode(encoder, [6, 4, 1], src_len=src_len)

        for kernel_width in [1, 3, 5]:
            for trg_len in [1, 2, 5, 9]:
                decoder = self._decoder(kernel_width=kernel_width)
                trg_embed = torch.rand(batch_size, trg_len, self.emb_size)
                trg_mask = torch.ones(batch_size, 1, trg_len, dtype=torch.bool)

                out, x, att, att_vectors = decoder(
                    trg_embed=trg_embed,
                    encoder_output=encoder_output,
                    encoder_hidden=None,
                    src_mask=src_mask,
                    unroll_steps=trg_len,
                    hidden=None,
                    trg_mask=trg_mask,
                )

                msg = f"k={kernel_width}, trg_len={trg_len}"
                self.assertIsNone(att_vectors, msg=msg)
                self.assertEqual(
                    out.shape,
                    torch.Size([batch_size, trg_len, self.vocab_size]),
                    msg=msg,
                )
                self.assertEqual(
                    x.shape,
                    torch.Size([batch_size, trg_len, self.hidden_size]),
                    msg=msg,
                )
                self.assertEqual(
                    att.shape, torch.Size([batch_size, trg_len, src_len]), msg=msg
                )
                self.assertEqual(decoder.output_size, self.vocab_size)
                self.assertTrue(torch.isfinite(out).all(), msg=msg)

    def test_conv_decoder_dummy_trg_mask(self):
        """The (1, 1, 1) placeholder trg_mask that search passes must work."""
        encoder = self._encoder()
        encoder_output, src_mask = self._encode(encoder, [6, 4])
        decoder = self._decoder()
        trg_embed = torch.rand(2, 5, self.emb_size)

        full_mask = torch.ones(2, 1, 5, dtype=torch.bool)
        out_full, _, _, _ = decoder(
            trg_embed, encoder_output, None, src_mask, 5, None, full_mask
        )
        out_dummy, _, _, _ = decoder(
            trg_embed,
            encoder_output,
            None,
            src_mask,
            5,
            None,
            src_mask.new_ones([1, 1, 1]),
        )
        torch.testing.assert_close(out_full, out_dummy)

    def test_conv_decoder_causality(self):
        """
        Output at position i must have exactly zero gradient w.r.t. trg_embed at
        every position j > i. This is the property `trg_mask` cannot give us.
        """
        trg_len = 8
        encoder = self._encoder()
        encoder_output, src_mask = self._encode(encoder, [6])

        for kernel_width in [3, 5]:
            decoder = self._decoder(kernel_width=kernel_width)
            for i in range(trg_len):
                trg_embed = torch.rand(1, trg_len, self.emb_size, requires_grad=True)
                out, _, _, _ = decoder(
                    trg_embed,
                    encoder_output,
                    None,
                    src_mask,
                    trg_len,
                    None,
                    torch.ones(1, 1, trg_len, dtype=torch.bool),
                )
                grad, = torch.autograd.grad(out[0, i, :].sum(), trg_embed)

                leaked = float(grad[0, i + 1:, :].abs().sum())
                self.assertEqual(
                    leaked,
                    0.0,
                    msg=f"k={kernel_width}: output {i} sees the future "
                    f"(leaked gradient {leaked})",
                )
                # and it really does depend on its own position
                self.assertGreater(
                    float(grad[0, i, :].abs().sum()),
                    0.0,
                    msg=f"k={kernel_width}: output {i} ignores its own input",
                )

    def test_conv_decoder_attention_rows_sum_to_one(self):
        """Attention is a distribution over the real source positions only."""
        src_lengths = [6, 4, 1]
        batch_size, trg_len = len(src_lengths), 5
        encoder = self._encoder()
        encoder_output, src_mask = self._encode(encoder, src_lengths, src_len=6)
        decoder = self._decoder()

        trg_embed = torch.rand(batch_size, trg_len, self.emb_size)
        _, _, _, _ = decoder(
            trg_embed,
            encoder_output,
            None,
            src_mask,
            trg_len,
            None,
            torch.ones(batch_size, 1, trg_len, dtype=torch.bool),
        )

        for layer_idx, attention in enumerate(decoder.layer_attentions):
            msg = f"layer {layer_idx}"
            torch.testing.assert_close(
                attention.sum(dim=-1),
                torch.ones(batch_size, trg_len),
                rtol=1e-5,
                atol=1e-6,
                msg=msg,
            )
            for i, length in enumerate(src_lengths):
                self.assertTrue(
                    torch.equal(
                        attention[i, :, length:],
                        torch.zeros(trg_len, 6 - length),
                    ),
                    msg=f"{msg}: probability mass on padded source positions",
                )

    def test_conv_decoder_per_layer_attention(self):
        """One attention map per layer, detached, matching the returned last one."""
        for num_layers in [1, 3, 6]:
            encoder = self._encoder()
            encoder_output, src_mask = self._encode(encoder, [6, 4])
            decoder = self._decoder(num_layers=num_layers)

            trg_embed = torch.rand(2, 5, self.emb_size)
            _, _, att, _ = decoder(
                trg_embed,
                encoder_output,
                None,
                src_mask,
                5,
                None,
                torch.ones(2, 1, 5, dtype=torch.bool),
            )

            self.assertEqual(len(decoder.layer_attentions), num_layers)
            self.assertEqual(len(decoder.attentions), num_layers)
            for attention in decoder.layer_attentions:
                self.assertEqual(attention.shape, torch.Size([2, 5, 6]))
                self.assertFalse(attention.requires_grad)
            # the returned attention is the last layer's
            torch.testing.assert_close(decoder.layer_attentions[-1], att.detach())
            # a second forward pass does not accumulate
            _ = decoder(
                trg_embed,
                encoder_output,
                None,
                src_mask,
                5,
                None,
                torch.ones(2, 1, 5, dtype=torch.bool),
            )
            self.assertEqual(len(decoder.layer_attentions), num_layers)


if __name__ == "__main__":
    unittest.main()
