import math
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from joeynmt.decoders import ConvDecoder
from joeynmt.encoders import ConvEncoder
from joeynmt.model import build_model
from joeynmt.vocabulary import Vocabulary


class TestConvModelInit(unittest.TestCase):
    """
    `initialize_model` runs after every module's __init__ and overwrites all
    weights, so `build_model` re-applies the ConvS2S init (SPEC.md §5) on top.
    These tests check that the weights a built model ends up with are the
    ConvS2S ones and not JoeyNMT's generic xavier_uniform.
    """

    def setUp(self):
        self.seed = 42
        tokens = [f"tok{i:03d}" for i in range(200)]
        special_symbols = SimpleNamespace(
            **{
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "sep_token": "<sep>",
                "unk_id": 0,
                "pad_id": 1,
                "bos_id": 2,
                "eos_id": 3,
                "sep_id": 4,
                "lang_tags": ["<de>", "<en>"],
            }
        )
        self.vocab = Vocabulary(tokens=tokens, cfg=special_symbols)

        self.emb_size = 32
        self.hidden_size = 64
        self.kernel_width = 3
        self.dropout = 0.1
        self.num_layers = 2

    def _cfg(self, tied_softmax=False):
        block = {
            "type": "conv",
            "hidden_size": self.hidden_size,
            "embeddings": {"embedding_dim": self.emb_size},
            "num_layers": self.num_layers,
            "kernel_width": self.kernel_width,
            "dropout": self.dropout,
        }
        return {
            "initializer": "xavier_uniform",
            "embed_initializer": "xavier_uniform",
            "bias_initializer": "zeros",
            "tied_embeddings": False,
            "tied_softmax": tied_softmax,
            "encoder": dict(block),
            "decoder": dict(block),
        }

    def _build(self, tied_softmax=False):
        torch.manual_seed(self.seed)
        return build_model(
            self._cfg(tied_softmax=tied_softmax),
            src_vocab=self.vocab,
            trg_vocab=self.vocab,
        )

    def test_conv_dispatch(self):
        model = self._build()
        self.assertIsInstance(model.encoder, ConvEncoder)
        self.assertIsInstance(model.decoder, ConvDecoder)

    def test_glu_conv_std_is_convs2s_not_xavier(self):
        """
        Every GLU conv weight must come out of N(0, sqrt(4p/n)), n = H * k.
        xavier_uniform on the same tensor gives a clearly different std, so a
        model that had only been through `initialize_model` would fail this.
        """
        model = self._build()

        fan_in = self.hidden_size * self.kernel_width
        expected = math.sqrt(4.0 * (1.0 - self.dropout) / fan_in)

        # what xavier_uniform would have produced on the same shape
        reference = torch.empty(
            2 * self.hidden_size, self.hidden_size, self.kernel_width
        )
        nn.init.xavier_uniform_(reference, gain=1.0)
        xavier_std = float(reference.std())

        # the two hypotheses must be far enough apart for this test to mean
        # anything at all
        self.assertGreater(abs(expected - xavier_std) / expected, 0.5)

        convs = [(f"encoder.layers.{i}", b.conv)
                 for i, b in enumerate(model.encoder.layers)]
        convs += [(f"decoder.layers.{i}", b.conv)
                  for i, b in enumerate(model.decoder.layers)]
        self.assertEqual(len(convs), 2 * self.num_layers)

        for name, conv in convs:
            std = float(conv.weight.std())
            self.assertAlmostEqual(
                std / expected,
                1.0,
                delta=0.05,
                msg=f"{name}: std {std:.5f} != sqrt(4p/n) {expected:.5f}",
            )
            self.assertGreater(
                abs(std - xavier_std) / xavier_std,
                0.3,
                msg=f"{name}: std {std:.5f} looks like xavier_uniform "
                f"{xavier_std:.5f}; the ConvS2S init did not survive",
            )
            # conv biases are zeroed by the ConvS2S init
            self.assertEqual(float(conv.bias.abs().sum()), 0.0, msg=name)

    def test_non_glu_layer_std(self):
        """Projections that do not feed a GLU use N(0, sqrt(1/n))."""
        model = self._build()
        for name, layer, fan_in in [
            ("encoder.input_proj", model.encoder.input_proj, self.emb_size),
            ("encoder.output_proj", model.encoder.output_proj, self.hidden_size),
            (
                "decoder.attentions.0.query_proj",
                model.decoder.attentions[0].query_proj, self.hidden_size
            ),
            (
                "decoder.attentions.0.out_proj", model.decoder.attentions[0].out_proj,
                self.emb_size
            ),
        ]:
            expected = math.sqrt(1.0 / fan_in)
            std = float(layer.weight.std())
            self.assertAlmostEqual(
                std / expected,
                1.0,
                delta=0.12,
                msg=f"{name}: std {std:.5f} != sqrt(1/n) {expected:.5f}",
            )

    def test_tied_softmax_survives_reset_parameters(self):
        """
        With tied_softmax the output layer's weight IS the target embedding
        table. `reset_parameters` must leave it alone, or it would overwrite the
        embedding init in place and un-zero the padding row.
        """
        model = self._build(tied_softmax=True)
        pad_index = self.vocab.pad_index

        self.assertIs(model.decoder.output_layer.weight, model.trg_embed.lut.weight)
        self.assertIsNone(model.decoder.output_layer.bias)
        self.assertEqual(
            float(model.trg_embed.lut.weight[pad_index].abs().sum()),
            0.0,
            msg="padding row was overwritten, so the tied weight got re-initialized",
        )

        # the tied weight must NOT look like the conv output-layer init
        conv_std = math.sqrt(1.0 / self.emb_size)
        std = float(model.trg_embed.lut.weight.std())
        self.assertGreater(
            abs(std - conv_std) / conv_std,
            0.3,
            msg=f"tied weight std {std:.5f} looks like sqrt(1/E) {conv_std:.5f}",
        )

        # teeth: without the guard, the pad row is destroyed
        model.decoder.reset_parameters(init_output_layer=True)
        self.assertGreater(
            float(model.trg_embed.lut.weight[pad_index].abs().sum()),
            0.0,
            msg="unguarded reset_parameters left the tied weight untouched; "
            "the guard in build_model proves nothing",
        )

    def test_untied_output_layer_is_initialized(self):
        """Without tying, the output layer does get the ConvS2S init."""
        model = self._build(tied_softmax=False)
        self.assertIsNot(model.decoder.output_layer.weight, model.trg_embed.lut.weight)
        expected = math.sqrt(1.0 / self.emb_size)
        std = float(model.decoder.output_layer.weight.std())
        self.assertAlmostEqual(std / expected, 1.0, delta=0.12)


if __name__ == "__main__":
    unittest.main()
