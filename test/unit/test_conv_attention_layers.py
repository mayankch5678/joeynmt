import unittest
from types import SimpleNamespace

import torch

from joeynmt.decoders import ConvDecoder
from joeynmt.encoders import ConvEncoder
from joeynmt.model import build_model
from joeynmt.vocabulary import Vocabulary


class TestConvAttentionLayers(unittest.TestCase):
    """Ablation switch `ConvDecoder(attention_layers="all" | "last")`."""

    def setUp(self):
        self.emb_size = 8
        self.hidden_size = 16
        self.kernel_width = 3
        self.vocab_size = 20
        self.seed = 42
        torch.manual_seed(self.seed)

    def _decoder(self, **overrides):
        cfg = {
            "hidden_size": self.hidden_size,
            "emb_size": self.emb_size,
            "num_layers": 3,
            "kernel_width": self.kernel_width,
            "dropout": 0.0,
            "emb_dropout": 0.0,
            "vocab_size": self.vocab_size,
            "max_position": 64,
        }
        cfg.update(overrides)
        torch.manual_seed(self.seed)
        decoder = ConvDecoder(**cfg)
        decoder.eval()
        return decoder

    def _encode(self, src_lengths):
        encoder = ConvEncoder(
            hidden_size=self.hidden_size,
            emb_size=self.emb_size,
            num_layers=2,
            kernel_width=self.kernel_width,
            dropout=0.0,
            emb_dropout=0.0,
            max_position=64,
        )
        encoder.eval()
        src_len = max(src_lengths)
        src_mask = torch.zeros(len(src_lengths), 1, src_len, dtype=torch.bool)
        for i, length in enumerate(src_lengths):
            src_mask[i, 0, :length] = True
        encoder_output, _ = encoder(
            torch.rand(len(src_lengths), src_len, self.emb_size),
            torch.tensor(src_lengths),
            src_mask,
        )
        return encoder_output, src_mask

    def _run(self, decoder, encoder_output, src_mask, trg_embed):
        batch, trg_len, _ = trg_embed.size()
        return decoder(
            trg_embed,
            encoder_output,
            None,
            src_mask,
            trg_len,
            None,
            torch.ones(batch, 1, trg_len, dtype=torch.bool),
        )

    def test_layer_attentions_length(self):
        """"last" -> one map, "all" -> one per layer."""
        encoder_output, src_mask = self._encode([6, 4])
        trg_embed = torch.rand(2, 5, self.emb_size)
        for num_layers in [1, 3, 6]:
            with self.subTest(num_layers=num_layers):
                last = self._decoder(num_layers=num_layers, attention_layers="last")
                _, _, att, _ = self._run(last, encoder_output, src_mask, trg_embed)
                self.assertEqual(len(last.layer_attentions), 1)
                self.assertEqual(len(last.attentions), 1)
                self.assertEqual(len(last.layers), num_layers)
                self.assertEqual(last.layer_attentions[0].shape, torch.Size([2, 5, 6]))
                torch.testing.assert_close(last.layer_attentions[0], att.detach())

                full = self._decoder(num_layers=num_layers, attention_layers="all")
                self._run(full, encoder_output, src_mask, trg_embed)
                self.assertEqual(len(full.layer_attentions), num_layers)
                self.assertEqual(len(full.attentions), num_layers)

                # and the default is "all"
                default = self._decoder(num_layers=num_layers)
                self.assertEqual(default.attention_layers, "all")
                self.assertEqual(len(default.attentions), num_layers)

    def test_last_does_not_accumulate(self):
        encoder_output, src_mask = self._encode([5])
        decoder = self._decoder(attention_layers="last")
        trg_embed = torch.rand(1, 4, self.emb_size)
        for _ in range(3):
            self._run(decoder, encoder_output, src_mask, trg_embed)
        self.assertEqual(len(decoder.layer_attentions), 1)

    def test_invalid_value_raises(self):
        with self.assertRaises(ValueError):
            self._decoder(attention_layers="first")

    def test_parameter_count_shrinks_by_the_dropped_attentions(self):
        num_layers = 4
        full = self._decoder(num_layers=num_layers, attention_layers="all")
        last = self._decoder(num_layers=num_layers, attention_layers="last")
        per_attention = sum(p.numel() for p in full.attentions[0].parameters())
        self.assertGreater(per_attention, 0)
        self.assertEqual(
            sum(p.numel() for p in full.parameters())
            - sum(p.numel() for p in last.parameters()),
            (num_layers - 1) * per_attention,
        )

    def test_last_is_still_causal(self):
        """Dropping the early attentions must not open a path to the future."""
        trg_len = 8
        encoder_output, src_mask = self._encode([6])
        decoder = self._decoder(attention_layers="last")
        for i in range(trg_len):
            trg_embed = torch.rand(1, trg_len, self.emb_size, requires_grad=True)
            out, _, _, _ = self._run(decoder, encoder_output, src_mask, trg_embed)
            grad, = torch.autograd.grad(out[0, i, :].sum(), trg_embed)
            self.assertEqual(float(grad[0, i + 1:, :].abs().sum()), 0.0)
            self.assertGreater(float(grad[0, i, :].abs().sum()), 0.0)

    def test_last_layer_attention_reaches_the_encoder(self):
        """With "last" the encoder is still trained, through the final layer."""
        encoder_output, src_mask = self._encode([6])
        encoder_output = encoder_output.detach().requires_grad_(True)
        decoder = self._decoder(attention_layers="last")
        out, _, _, _ = self._run(
            decoder, encoder_output, src_mask, torch.rand(1, 4, self.emb_size)
        )
        grad, = torch.autograd.grad(out.sum(), encoder_output)
        self.assertGreater(float(grad.abs().sum()), 0.0)


class TestConvAttentionLayersBuildModel(unittest.TestCase):
    """`build_model` sets the encoder's gradient scaling from the attention count."""

    def setUp(self):
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
        self.vocab = Vocabulary(
            tokens=[f"tok{i:03d}" for i in range(50)], cfg=special_symbols
        )
        self.num_layers = 3

    def _build(self, **decoder_overrides):
        block = {
            "type": "conv",
            "hidden_size": 16,
            "embeddings": {"embedding_dim": 8},
            "num_layers": self.num_layers,
            "kernel_width": 3,
            "dropout": 0.1,
        }
        cfg = {
            "initializer": "xavier_uniform",
            "embed_initializer": "xavier_uniform",
            "bias_initializer": "zeros",
            "tied_embeddings": False,
            "tied_softmax": False,
            "encoder": dict(block),
            "decoder": {**block, **decoder_overrides},
        }
        torch.manual_seed(42)
        return build_model(cfg, src_vocab=self.vocab, trg_vocab=self.vocab)

    def test_num_attention_layers(self):
        self.assertEqual(self._build().encoder.num_attention_layers, self.num_layers)
        self.assertEqual(
            self._build(attention_layers="all").encoder.num_attention_layers,
            self.num_layers,
        )
        last = self._build(attention_layers="last")
        self.assertEqual(last.encoder.num_attention_layers, 1)
        self.assertEqual(len(last.decoder.layers), self.num_layers)

    def test_encoder_gradient_scale_follows(self):
        """1/L_d for "all", exactly unscaled (1/1) for "last"."""
        grads = {}
        for mode in ["all", "last"]:
            model = self._build(attention_layers=mode)
            model.eval()
            src_embed = torch.rand(1, 5, 8, requires_grad=True)
            mask = torch.ones(1, 1, 5, dtype=torch.bool)
            out, _ = model.encoder(src_embed, torch.tensor([5]), mask)
            grads[mode], = torch.autograd.grad(out.sum(), src_embed)
            # unscaled reference: same encoder with the hook switched off
            model.encoder.num_attention_layers = None
            out, _ = model.encoder(src_embed, torch.tensor([5]), mask)
            reference, = torch.autograd.grad(out.sum(), src_embed)
            expected = 1.0 / (self.num_layers if mode == "all" else 1)
            torch.testing.assert_close(grads[mode], reference * expected)


if __name__ == "__main__":
    unittest.main()
