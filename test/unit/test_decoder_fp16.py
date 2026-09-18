import unittest

import torch

from joeynmt.decoders import RecurrentDecoder
from joeynmt.encoders import RecurrentEncoder


class TestRecurrentDecoderFp16InitHidden(unittest.TestCase):
    """
    `search.py:411` calls `RecurrentDecoder._init_hidden` outside the autocast
    context, so with fp16 the encoder state arrives as Half while the bridge
    layer's weights are still Float. Only `init_hidden: "bridge"` does a matmul
    there, so only that path can raise.
    """

    def setUp(self):
        self.seed = 42
        self.emb_size = 8
        self.hidden_size = 16
        self.vocab_size = 20
        self.batch_size = 3

    def _decoder(self, init_hidden="bridge"):
        torch.manual_seed(self.seed)
        encoder = RecurrentEncoder(
            rnn_type="gru",
            hidden_size=self.hidden_size,
            emb_size=self.emb_size,
            num_layers=1,
            bidirectional=True,
        )
        decoder = RecurrentDecoder(
            rnn_type="gru",
            emb_size=self.emb_size,
            hidden_size=self.hidden_size,
            encoder=encoder,
            attention="luong",
            num_layers=1,
            vocab_size=self.vocab_size,
            init_hidden=init_hidden,
        )
        decoder.eval()
        return encoder, decoder

    def test_bridge_accepts_half_encoder_state(self):
        """The fp16 beam-search path: Half in, Float weights, must not raise."""
        _, decoder = self._decoder("bridge")
        encoder_final = torch.randn(
            self.batch_size, 2 * self.hidden_size, dtype=torch.float16
        )
        hidden = decoder._init_hidden(encoder_final)  # pylint: disable=protected-access

        self.assertEqual(
            hidden.shape, torch.Size([1, self.batch_size, self.hidden_size])
        )
        self.assertEqual(hidden.dtype, decoder.bridge_layer.weight.dtype)
        self.assertTrue(torch.isfinite(hidden).all())

    def test_fp32_behaviour_unchanged(self):
        """
        In fp32 the cast is a no-op: same values, and `.to()` returns the very
        same tensor object rather than a copy.
        """
        _, decoder = self._decoder("bridge")
        encoder_final = torch.randn(self.batch_size, 2 * self.hidden_size)
        self.assertIs(
            encoder_final.to(decoder.bridge_layer.weight.dtype), encoder_final
        )

        torch.manual_seed(self.seed)
        expected = (
            decoder.activation(decoder.bridge_layer(encoder_final)
                               ).unsqueeze(0).repeat(decoder.num_layers, 1, 1)
        )
        hidden = decoder._init_hidden(encoder_final)  # pylint: disable=protected-access
        torch.testing.assert_close(hidden, expected)

    def test_half_and_float_agree(self):
        """The Half path must give the same answer as casting up front."""
        _, decoder = self._decoder("bridge")
        encoder_final = torch.randn(self.batch_size, 2 * self.hidden_size)
        # pylint: disable=protected-access
        from_float = decoder._init_hidden(encoder_final)
        from_half = decoder._init_hidden(encoder_final.half())
        torch.testing.assert_close(from_half, from_float, rtol=1e-2, atol=1e-2)

    def test_other_init_hidden_options_accept_half(self):
        """`last` and `zero` do no matmul, so they were never affected."""
        for option in ("last", "zero"):
            _, decoder = self._decoder(option)
            encoder_final = torch.randn(
                self.batch_size, 2 * self.hidden_size, dtype=torch.float16
            )
            # pylint: disable=protected-access
            hidden = decoder._init_hidden(encoder_final)
            self.assertEqual(
                hidden.shape,
                torch.Size([1, self.batch_size, self.hidden_size]),
                msg=option,
            )


if __name__ == "__main__":
    unittest.main()
