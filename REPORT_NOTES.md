# Report notes

Running log of everything the final report will need.
Append as work happens. Do not reconstruct at the end.

## Design decisions and why
### Decoder incremental state: not needed (2026-09-17)
JoeyNMT's `TransformerDecoder` re-runs the full target prefix at every decoding
step; there is no KV cache or incremental state anywhere (`search.py:160`,
`search.py:508-532`). Therefore the conv decoder needs no incremental convolution
buffers: the same `left-pad(k-1)` + `trim(k-1)` code is correct at both training
and inference. Cost is O(T^2) decoder passes, the same as the existing Transformer
baseline, so the comparison stays fair. We explicitly give up ConvS2S's
inference-speed advantage; discuss in the report.

### search.py must learn about ConvDecoder (2026-09-17)
`search.py` dispatches on `isinstance` in three places (`search.py:45-61`,
`search.py:342`, `search.py:405-418`). `ConvDecoder` must be added to the
Transformer-style branches or decoding raises `NotImplementedError`. This is the
only modification to existing JoeyNMT decoding logic.

### Causality cannot come from trg_mask (2026-09-17)
`transformer_greedy` passes `trg_mask = src_mask.new_ones([1, 1, 1])`, a dummy
placeholder. Causality therefore cannot come from `trg_mask`; it must come from
left-padding and trimming alone. This makes a gradient-based causality test
essential.

### ConvEncoder returns keys and values in one tensor (2026-09-17)
`ConvEncoder.forward` returns `(cat([z_u, z_c], -1), None)`, shape `(B, S, 2E)`,
not `(B, S, H)`. Multi-step attention needs both the keys `z_u` and the values
`z_c = sqrt(0.5)(z_u + e)`, and the `encoder_hidden` slot cannot carry the second
one: `search.py` hardcodes `encoder_hidden=None` on the Transformer decode path
(`search.py:242, 518, 543`), and `encoder_output` is the only tensor beam search
tiles by `k` and re-indexes with `select_indices`. Stashing `z_c` on the module
would survive greedy decoding but not beam reordering.
`encoder.output_size` stays `hidden_size` (what the Noam scheduler reads at
`training.py:121`); `encoder.attention_size` is the width of each half.

### Weight normalisation via torch parametrizations (2026-09-17)
`torch.nn.utils.weight_norm` is deprecated in torch 2.1.2 and raises a
`UserWarning`. `conv_layers.py` uses `torch.nn.utils.parametrizations.weight_norm`
instead. Re-initialising a wrapped conv is done by assigning
`module.weight = new_tensor` under `no_grad`; the parametrization's
`right_inverse` recomputes `g = ||v||` so the effective weight equals the assigned
tensor exactly (verified by round-trip). This keeps parameter identity stable, so
`reset_parameters()` can run after `initialize_model` without rebuilding modules.

### Dropout sits on the conv input, not after the GLU (2026-09-17)
Matches the paper and fairseq, and is what the `N(0, sqrt(4p/n))` init assumes:
`p` is the retain probability of the dropout applied to that conv's input.

### Projections are conditional (2026-09-17)
`input_proj`/`output_proj` exist only when `emb_size != hidden_size`; with
`E == H` the ConvS2S `fc1`/`fc2` become the identity and are omitted entirely.

### Attention sits inside the decoder conv block, so the block cannot close
its own residual (2026-09-17)
SPEC.md (D15)-(D16) are two nested residuals: `o = sqrt(0.5)(h~ + c_h)` and then
`h = sqrt(0.5)(o + h_prev)`. The attention therefore has to run between the GLU
and the block residual. `GLUConvBlock` gained a `residual: bool = True` switch:
the encoder keeps the default (E9), the decoder passes `residual=False` to get
the bare GLU output and performs (D16) itself. Default unchanged, so the encoder
and its tests are untouched.

### Multi-step attention scale is sqrt(m), not m (2026-09-17)
The paper's `c <- c * m * sqrt(1/m)` simplifies to `c * sqrt(m)`, with `m` the
number of *real* (non-pad) source tokens per sentence, i.e.
`src_mask.sum(-1)`, not the padded `src_len`. Using the padded length would make
the scale depend on the rest of the batch and break batch invariance.

### Per-layer attention is stored detached (2026-09-17)
`ConvDecoder.layer_attentions` holds `L_d` maps of shape (B, T, S) for the report
figures. They are `.detach()`ed: storing live tensors on `self` would keep the
autograd graph alive between the backward pass and the next forward. The list is
rebuilt at the start of every forward, so it never accumulates.

### Weight normalisation: convolutions only (2026-09-17)
fairseq's reference fconv weight-normalises every `Linear` as well as every conv.
Here it is applied to the GLU convolutions only, where the `N(0, sqrt(4p/n))`
variance argument bites; the six projections (`input_proj`, `output_proj`,
attention `query_proj`/`out_proj`, `output_layer`) are plain `nn.Linear`.
Deviation to revisit if training is unstable; a one-line change in each
`reset_parameters`.

### Only additive change to a baseline file so far (2026-09-17)
`joeynmt/decoders.py` needed `List` added to its `typing` import for the
`layer_attentions` annotation. No existing class was touched: `git diff` on
`encoders.py` and `decoders.py` is 367 insertions with that single import line as
the only replacement.

### tied_softmax and ConvDecoder.output_layer (2026-09-18)
`model.py:418-426` is the whole of softmax tying. It compares
`trg_embed.lut.weight.shape` with `model.decoder.output_layer.weight.shape` and,
on a match, assigns `output_layer.weight = trg_embed.lut.weight`. It never looks
at `.bias`, so `bias=False` is a modelling convention, not a requirement.
On a shape mismatch it does **not** no-op: it raises `ConfigurationError`
("the decoder embedding_dim and decoder hidden_size must be the same. The decoder
must be a Transformer"), so a mis-sized decoder fails loudly at build time.
`ConvDecoder.output_layer` is `nn.Linear(emb_size, vocab_size, bias=False)`, i.e.
weight `(V, E)`, and `build_model` injects `emb_size=trg_embed.embedding_dim` into
every decoder branch, so the shapes agree by construction for any `hidden_size`.
Tying therefore works for the conv decoder without the `E == H` constraint the
Transformer has; `configs/conv_small.yaml` exercises exactly that, with
`embedding_dim: 64` and `hidden_size: 128` and `tied_softmax: True`.
Verified after `build_model`: `output_layer.weight is trg_embed.lut.weight` is
True, both `(204, 64)`, `output_layer.bias is None`.

### Pipeline wiring, minimal diff (2026-09-18)
Five lines replaced in two baseline files, nothing else:
- `model.py`: two import lines; an `elif enc_cfg.get("type", "recurrent") ==
  "conv"` before the recurrent fallback, and the same for the decoder before its
  fallback. Order matters: the existing code is an if/else, so without the elif
  `type: conv` silently built an RNN.
- `model.py`: after both are built,
  `encoder.num_attention_layers = len(decoder.layers)`, guarded by an
  `isinstance` check on both. This is what turns on the encoder gradient scaling
  (E15). It is the *decoder's* layer count because every conv decoder layer
  attends.
- `search.py`: one import line, plus `(TransformerDecoder, ConvDecoder)` in the
  `greedy` dispatch (line 45) and in `is_transformer` (line 398). The
  `RecurrentDecoder` branch at line 54 is untouched, so RNN and Transformer
  decoding are bit-identical. `is_transformer` now really means "full-prefix
  decoder, no incremental state"; kept the name to keep the diff minimal and
  added a comment.

### ConvS2S init is currently NOT applied end to end (2026-09-18)
`ConvEncoder.reset_parameters()` / `ConvDecoder.reset_parameters()` run in
`__init__`, but `build_model` calls `initialize_model` afterwards
(`model.py:429`), which overwrites every weight with `xavier_uniform`. So the
trained toy model uses JoeyNMT's generic init, not SPEC.md §5.
Re-running `reset_parameters()` after `initialize_model` needs a guard: with
`tied_softmax` the output layer's weight *is* the target embedding table, and
`init_default_layer_` copies in place, so it would clobber the embeddings'
`N(0, 0.1)` init and un-zero their pad row. Not fixed yet; deliberately left out
of the wiring diff.

### initialize_model crashes on non-transformer + xavier_normal (2026-09-18)
Pre-existing baseline bug, not ours: `initialization.py:136-139` only defines
`deepnet` when both types are `"transformer"`, but line 194 reads
`init in deepnet` whenever `init == "xavier_normal"`. Any non-transformer model
(RNN too) with `initializer: xavier_normal` raises `NameError`.
`configs/conv_small.yaml` uses `xavier_uniform`, so it is unaffected.

## Paper ambiguities and resolutions
Architecture frozen as equations and tensor shapes in `SPEC.md` (2026-09-17).
`SPEC.md` §6 lists the five ambiguities and the chosen reading:
attention-query scaling, non-GLU init variance, the sqrt(0.5) on `z + e`,
learned positional tables, and which layer's attention is returned.

## Correctness evidence

### ConvEncoder, `test/unit/test_conv_encoder.py` (2026-09-17), 7 tests, all pass
- `test_conv_encoder_output_shape`: over k in {1,3,5} and S in {1,2,5,11}, output
  is `(B, S, 2E)` and finite. Proves symmetric padding preserves length for every
  kernel width, including S < k.
- `test_conv_encoder_padding_invariance`: a sentence encoded alone equals the same
  sentence inside a padded batch at its real positions (`assert_close`,
  atol 1e-6), with random garbage in the padded slots; the padded tail is exactly
  zero. Proves no pad position leaks into a real position.
  Mutation check: disabling the per-block mask leaves sentence 0 (the longest,
  unpadded) unchanged but breaks sentences 1 and 2 by 0.58 and 0.61 max abs diff,
  i.e. ~5 orders of magnitude above tolerance. The test has teeth.
- `test_conv_encoder_parameters_and_gradients`: parameter count equals the
  closed-form architecture count (pos table + projections + per-layer
  `2Hk*H + 2H` weight-norm v/g + `2H` bias), all parameters finite after
  ConvS2S init, and after `output.sum().backward()` every leaf has a finite,
  non-zero gradient.
- `test_conv_encoder_gradient_scaling`: with `num_attention_layers = 4` the
  gradient reaching `src_embed` is exactly 1/4 of the unscaled one.
- `test_conv_encoder_freeze`: `freeze=True` clears `requires_grad` everywhere.
- `test_glu_conv_block_causality`: for k in {3,5} and N=8, backprop from
  `output[0, i, :].sum()` into the input for every i; the gradient at all
  positions j > i is exactly 0.0. This is the decisive test for the decoder
  block, because `trg_mask` cannot supply causality (search passes `[1,1,1]`);
  it comes from left-pad(k-1) + trim(k-1) alone.
- `test_glu_conv_block_causality_test_has_teeth`: the identical check run with
  `padding_mode="symmetric"` must find a strictly positive future gradient at
  every position but the last. It does, so the causality assertion above is not
  vacuous (e.g. it would not pass merely because the gradient path were broken).

Full suite after this component: 84 tests, `OK (skipped=1)` (was 77).

### ConvDecoder, `test/unit/test_conv_decoder.py` (2026-09-17), 6 tests, all pass
- `test_conv_decoder_output_shape`: over k in {1,3,5} and T in {1,2,5,9}, returns
  logits (B,T,V), states (B,T,H), attention (B,T,S), `None` in the att_vectors
  slot, and `output_size == vocab_size`.
- `test_conv_decoder_causality`: for k in {3,5} and T=8, backprop from
  `out[0, i, :].sum()` into `trg_embed`; gradient at every j > i is exactly 0.0,
  and the gradient at j = i is strictly positive so the zero is not vacuous.
  This is the decisive correctness test: causality comes from left-pad(k-1) +
  trim(k-1) alone, never from `trg_mask`.
- `test_conv_decoder_dummy_trg_mask`: the `(1,1,1)` placeholder `trg_mask` that
  `search.py` passes gives bit-identical logits to a real `(B,1,T)` mask on an
  unpadded batch. Proves the decoder does not depend on `trg_mask` for anything
  but zeroing pads.
- `test_conv_decoder_attention_rows_sum_to_one`: in every layer, attention rows
  sum to 1.0 and put exactly zero mass on padded source positions, with source
  lengths [6,4,1] in one batch.
- `test_conv_decoder_per_layer_attention`: for L_d in {1,3,6},
  `len(layer_attentions) == L_d == len(attentions)`, every map is (B,T,S) and
  detached, the returned `att` equals the last layer's, and a second forward does
  not accumulate.
- `test_conv_decoder_freeze`: `freeze=True` clears `requires_grad` everywhere.

Full suite after this component: 90 tests, `OK (skipped=1)` (was 84).

### End-to-end smoke run, `configs/conv_small.yaml` (2026-09-18)
`python -m joeynmt train configs/conv_small.yaml` on CPU, toy data, 32 updates,
1 epoch: completed without error. Training loss 2945.04 over 315 seqs / 6117
tokens in 0.69s; best dev loss 252.16 at step 30; greedy validation, then beam
search (beam 5) on dev and test, dev BLEU 0.12. The BLEU number is meaningless at
this scale -- what it proves is that the whole path works: conv dispatch in
`build_model`, training forward/backward, greedy decoding via
`transformer_greedy`, and beam search via the full-prefix branch.
Log confirms `encoder=ConvEncoder(num_layers=3, hidden_size=128, emb_size=64,
kernel_width=3, num_attention_layers=3)` and the matching `ConvDecoder`;
767,424 total params.

Note in the trainable-parameter list: `decoder.output_layer.weight` does not
appear. That is correct, not a freeze bug -- `named_parameters()` de-duplicates
tied tensors and yields the first registered name, and `Model.__init__` registers
`trg_embed` before `decoder` (`model.py:53-56`), so the shared weight surfaces as
`trg_embed.lut.weight`. The same de-duplication is why `initialize_model` gives
the tied weight the *embedding* initializer rather than the generic one.

Full suite after wiring: 90 tests, `OK (skipped=1)`, unchanged.
Planned:
- padding-invariance test for the encoder
- gradient-based causality test for the decoder
- overfit-50-sentences test for the full model

## Bugs and fixes

## Environment and reproducibility
- Local: conda env "convs2s", Python 3.11.16, torch 2.1.2, CPU only
- Upstream baseline commit: cdc4d03d430a1b0f29793a0d95743c5e72ae2f6c (2024-01-25)
- Pins required to run JoeyNMT v2.3 in 2026:
  numpy<2 (torch 2.1.2 NumPy 1.x ABI), sentencepiece==0.1.99
  (0.2.x drops SetVocabulary), importlib_metadata (missing from requirements)
- Baseline test suite: OK (skipped=1)
- Colab setup: TBD

## Results and observations

## Limitations and open questions

## References
- Gehring et al. 2017, Convolutional Sequence to Sequence Learning
- Dauphin et al. 2017, Language Modeling with Gated Convolutional Networks
- Kreutzer et al. 2019, Joey NMT: A Minimalist NMT Toolkit for Novices
