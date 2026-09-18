import unittest

import torch

from joeynmt.conv_layers import GLUConvBlock
from joeynmt.encoders import ConvEncoder


class TestConvEncoder(unittest.TestCase):

    def setUp(self):
        self.emb_size = 8
        self.hidden_size = 16
        self.num_layers = 3
        self.kernel_width = 3
        self.max_position = 64
        self.seed = 42
        torch.manual_seed(self.seed)

    def _encoder(self, **overrides):
        cfg = {
            "hidden_size": self.hidden_size,
            "emb_size": self.emb_size,
            "num_layers": self.num_layers,
            "kernel_width": self.kernel_width,
            "dropout": 0.0,
            "emb_dropout": 0.0,
            "max_position": self.max_position,
        }
        cfg.update(overrides)
        torch.manual_seed(self.seed)
        encoder = ConvEncoder(**cfg)
        encoder.eval()  # no dropout, deterministic
        return encoder

    def test_conv_encoder_freeze(self):
        encoder = self._encoder(freeze=True)
        for _, p in encoder.named_parameters():
            self.assertFalse(p.requires_grad)

    def test_conv_encoder_output_shape(self):
        """Output width is 2 * emb_size (keys ++ values) and length is preserved."""
        batch_size = 3
        for kernel_width in [1, 3, 5]:
            for src_len in [1, 2, 5, 11]:
                encoder = self._encoder(kernel_width=kernel_width)
                src_embed = torch.rand(batch_size, src_len, self.emb_size)
                src_length = torch.full((batch_size, ), src_len)
                mask = torch.ones(batch_size, 1, src_len, dtype=torch.bool)

                output, hidden = encoder(src_embed, src_length, mask)

                self.assertIsNone(hidden)
                self.assertEqual(
                    output.shape,
                    torch.Size([batch_size, src_len, 2 * self.emb_size]),
                    msg=f"k={kernel_width}, src_len={src_len}",
                )
                self.assertEqual(encoder.output_size, self.hidden_size)
                self.assertEqual(encoder.attention_size, self.emb_size)
                self.assertTrue(torch.isfinite(output).all())

    def test_conv_encoder_padding_invariance(self):
        """
        A sentence encoded alone must give exactly the same vectors as the same
        sentence sitting in a padded batch, at its real positions. Padded
        positions must come out zero.
        """
        lengths = [7, 4, 1]
        max_len = max(lengths)
        batch_size = len(lengths)
        encoder = self._encoder()

        # garbage in the padded slots: it must not reach the real positions
        padded_embed = torch.randn(batch_size, max_len, self.emb_size)
        mask = torch.zeros(batch_size, 1, max_len, dtype=torch.bool)
        for i, length in enumerate(lengths):
            mask[i, 0, :length] = True

        batched_output, _ = encoder(padded_embed, torch.tensor(lengths), mask)

        for i, length in enumerate(lengths):
            single_embed = padded_embed[i:i + 1, :length, :]
            single_output, _ = encoder(
                single_embed,
                torch.tensor([length]),
                torch.ones(1, 1, length, dtype=torch.bool),
            )
            torch.testing.assert_close(
                batched_output[i:i + 1, :length, :],
                single_output,
                rtol=1e-5,
                atol=1e-6,
                msg=f"padding invariance broken for sentence {i} (len {length})",
            )
            # padded tail is exactly zero
            self.assertTrue(
                torch.equal(
                    batched_output[i, length:, :],
                    torch.zeros(max_len - length, 2 * self.emb_size),
                )
            )

    def test_conv_encoder_parameters_and_gradients(self):
        """Parameter count matches the architecture and gradients reach every leaf."""
        encoder = self._encoder()

        k, h, e = self.kernel_width, self.hidden_size, self.emb_size
        expected = (
            self.max_position * e  # pos_embed
            + e * h + h  # input_proj
            + self.num_layers * (2 * h * h * k + 2 * h + 2 * h)  # conv v, g, bias
            + h * e + e  # output_proj
        )
        n_params = sum(p.numel() for p in encoder.parameters())
        self.assertEqual(n_params, expected)
        self.assertTrue(
            all(torch.isfinite(p).all() for p in encoder.parameters()),
            msg="non-finite parameter after ConvS2S init",
        )

        batch_size, src_len = 2, 6
        src_embed = torch.rand(batch_size, src_len, e, requires_grad=True)
        mask = torch.ones(batch_size, 1, src_len, dtype=torch.bool)
        mask[1, 0, 4:] = False

        output, _ = encoder(src_embed, torch.tensor([6, 4]), mask)
        output.sum().backward()

        self.assertIsNotNone(src_embed.grad)
        self.assertTrue(torch.isfinite(src_embed.grad).all())
        for name, p in encoder.named_parameters():
            self.assertIsNotNone(p.grad, msg=f"no gradient for {name}")
            self.assertTrue(torch.isfinite(p.grad).all(), msg=f"nan grad for {name}")
            if "pos_embed" not in name:
                self.assertGreater(
                    float(p.grad.abs().sum()), 0.0, msg=f"zero gradient for {name}"
                )

    def test_conv_encoder_gradient_scaling(self):
        """num_attention_layers divides the gradient reaching the encoder."""
        grads = {}
        for num_attention_layers in [None, 4]:
            encoder = self._encoder()
            encoder.num_attention_layers = num_attention_layers
            src_embed = torch.rand(2, 5, self.emb_size, requires_grad=True)
            mask = torch.ones(2, 1, 5, dtype=torch.bool)
            output, _ = encoder(src_embed, torch.tensor([5, 5]), mask)
            output.sum().backward()
            grads[num_attention_layers] = src_embed.grad.clone()

        torch.testing.assert_close(grads[4], grads[None] / 4.0)

    def _future_grad_norms(self, padding_mode, kernel_width, seq_len=8):
        """
        For every position i, return the total absolute gradient that
        output[0, i, :] sends back to the strictly future inputs j > i.
        """
        torch.manual_seed(self.seed)
        block = GLUConvBlock(
            hidden_size=self.hidden_size,
            kernel_width=kernel_width,
            dropout=0.0,
            padding_mode=padding_mode,
        )
        block.eval()

        future_grads = []
        for i in range(seq_len):
            x = torch.randn(1, seq_len, self.hidden_size, requires_grad=True)
            output = block(x)
            grad, = torch.autograd.grad(output[0, i, :].sum(), x)
            future_grads.append(float(grad[0, i + 1:, :].abs().sum()))
        return future_grads

    def test_glu_conv_block_causality(self):
        """Causal padding: output i must not depend on any input j > i."""
        for kernel_width in [3, 5]:
            future_grads = self._future_grad_norms("causal", kernel_width)
            for i, leaked in enumerate(future_grads):
                self.assertEqual(
                    leaked,
                    0.0,
                    msg=f"causal k={kernel_width}: output {i} sees the future "
                    f"(leaked gradient {leaked})",
                )

    def test_glu_conv_block_causality_test_has_teeth(self):
        """The same check must FAIL for symmetric padding, which is not causal."""
        for kernel_width in [3, 5]:
            future_grads = self._future_grad_norms("symmetric", kernel_width)
            # every position but the last one reads (k-1)/2 steps into the future
            for i, leaked in enumerate(future_grads[:-1]):
                self.assertGreater(
                    leaked,
                    0.0,
                    msg=f"symmetric k={kernel_width}: output {i} unexpectedly "
                    f"independent of the future; the causality test proves nothing",
                )


if __name__ == "__main__":
    unittest.main()
