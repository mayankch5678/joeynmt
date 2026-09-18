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

### ConvS2S init ordering, FIXED (2026-09-18)
Superseded the entry below. `build_model` now re-runs `reset_parameters()` on
both conv modules immediately after `initialize_model`, guarded by `isinstance`.
The `tied_softmax` case is handled with
`model.decoder.reset_parameters(init_output_layer=not tied)` where
`tied = model.decoder.output_layer.weight is trg_embed.lut.weight`: the init
writes in place, so re-initialising a tied output layer would overwrite the
target embedding table and un-zero its padding row.

Measured, with H=64, k=3, dropout=0.1 (so n=192, p=0.9, target std 0.13693):
- before the fix, a GLU conv weight came out at std 0.00937, ~15x too small.
  That is not xavier_uniform on the conv weight (0.0589) either: `initialize_model`
  sees the weight-norm parametrization's two tensors, `original1` (v, 3-dim) and
  `original0` (g, shape (2H,1,1)), and xavier-initialises each separately.
  fan_in=1 on the g tensor makes the reconstructed `weight = g * v / ||v||`
  badly scaled. So the generic init is not merely un-paper-like for a
  weight-normalised conv, it is broken.
- after the fix, std 0.13693 as intended.

### [superseded] ConvS2S init is currently NOT applied end to end (2026-09-18)
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

### Three-way comparison: matched parameter budgets (2026-09-18)
`configs/multi30k_{conv,rnn,transformer}.yaml`. Everything from `data:` to
`model:` is byte-identical in all three (verified: same md5 over that range), so
data, tokenizer, vocabulary, beam size 5, length penalty 1.0, sacrebleu
`tokenize: "13a"` and the whole optimisation block are held fixed. Only the
`model:` block differs.

The comparison is made independent of the final BPE vocabulary size by design:
all three use `embedding_dim: 256`, tie nothing, and end in an output layer of
width 256, so each carries exactly `3 * V * 256` vocabulary-dependent parameters
(2 embedding tables + output layer). Verified by counting at V=10000 and V=6000:
the `body` column is identical, only `shared` moves. This matters because the
vocabulary does not exist yet -- `scripts/get_multi30k.sh` has not been run --
and it means the sizing does not have to be redone afterwards.

To keep that property the RNN decoder's `hidden_size` is pinned to 256:
`RecurrentDecoder.output_layer` is `Linear(hidden_size, vocab_size)`, so a wider
decoder would have made its vocabulary-dependent share differ from the other two.
RNN capacity is tuned through the encoder instead.

Final counts at V=10000 (`python scripts/count_params.py configs/multi30k_*.yaml`):

| config | total | shared (3*V*256) | body | encoder | decoder |
|---|---|---|---|---|---|
| conv | 10,965,504 | 7,680,000 | 3,285,504 | 1,642,496 | 1,643,008 |
| transformer | 10,843,904 | 7,680,000 | 3,163,904 | 1,581,824 | 1,582,080 |
| rnn | 11,097,600 | 7,680,000 | 3,417,600 | 1,972,224 | 1,445,376 |

Spread: 8.0% over the architecture-specific body, 2.3% over the total. Within
the 10% budget.

Sizes chosen:
- conv: hidden 256, 4 encoder / 3 decoder layers, k=3. Unchanged (the anchor).
- transformer: hidden 256 (forced to equal `embedding_dim`), `ff_size` 512
  (2x hidden), 8 heads, 3 encoder / 2 decoder layers. The 3/2 split mirrors the
  conv model's deeper-encoder shape; a symmetric 3/3 at ff 512 is +20.4% and
  misses the budget.
- rnn: 2-layer bidirectional GRU encoder, hidden 256 (`encoder.output_size` 512),
  2-layer GRU decoder hidden 256, Luong attention, input feeding, bridge init.

Alternatives inside the budget, recorded in case the shapes need to change:
transformer (ff 384, 4 enc, 2 dec) is +0.4% vs conv; rnn (hidden 384, 1 enc,
2 dec) gives a 5.3% three-way spread instead of 8.0% but a shallower encoder.

### Bug in my own parameter counting, fixed before it mattered (2026-09-18)
The first sweep split "embedding" from "body" with a substring match on
`"output_layer" in name`. `MultiHeadedAttention` also owns a submodule called
`output_layer` (`transformer_layers.py`), so every Transformer attention output
projection was being counted as vocabulary-dependent. That understated the
Transformer body by 592,128 parameters and made a 3+3 / ff=512 Transformer look
like +2.3% when it is really +20.4%.
`scripts/count_params.py` now matches the three exact parameter names
(`src_embed.lut.weight`, `trg_embed.lut.weight`, `decoder.output_layer.weight`).
Worth remembering for the report's parameter table: never split JoeyNMT
parameters by substring.

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

### ConvS2S init after build_model, `test/unit/test_conv_model_init.py` (2026-09-18), 5 tests
- `test_conv_dispatch`: `type: conv` really builds `ConvEncoder`/`ConvDecoder`.
- `test_glu_conv_std_is_convs2s_not_xavier`: every GLU conv weight in a built
  model has std within 5% of `sqrt(4p/n)`, and at least 30% away from what
  `xavier_uniform` produces on the same shape. The test first asserts the two
  hypotheses are >50% apart, so it cannot pass by them being indistinguishable.
  Conv biases are exactly zero.
- `test_non_glu_layer_std`: the four projection types match `sqrt(1/n)`.
- `test_tied_softmax_survives_reset_parameters`: with `tied_softmax: True`,
  `output_layer.weight is trg_embed.lut.weight`, the pad row is still exactly
  zero, and the tied weight's std does not look like `sqrt(1/E)`. Teeth: calling
  `reset_parameters(init_output_layer=True)` by hand afterwards does destroy the
  pad row, so the guard is what protects it.
- `test_untied_output_layer_is_initialized`: without tying the output layer does
  get `sqrt(1/E)`, i.e. the guard is not just disabling the init everywhere.

Mutation check on the fix itself: disabling the two `reset_parameters()` calls in
`build_model` fails 3 of the 5 tests (GLU conv std 0.00937 vs 0.13693, non-GLU
projection 0.14470 vs 0.17678, untied output layer off by 48%).

Toy run re-verified after the fix: `python -m joeynmt train configs/conv_small.yaml`
completes, 767,424 params (unchanged, init does not alter shapes), training loss
2946.91, best dev loss 252.42, greedy and beam search both run.

Full suite: 95 tests, `OK (skipped=1)` (was 90).

### Overfit-50 test, `scripts/overfit_test.py` + `configs/conv_overfit.yaml` (2026-09-18)
First 50 pairs of `test/data/toy/train`, used as both train and dev, conv model
H=64, 2 layers, k=3, no dropout, Adam lr 1e-3, CPU, capped at 2000 steps.
PASS: per-token loss 0.0001 (target < 0.1), greedy BLEU 100.00 (target > 95),
teacher-forced accuracy 1.000. Curve:

| steps | loss/token | bleu | acc |
|---|---|---|---|
| 200 | 0.1277 | 81.30 | 0.992 |
| 400 | 0.0290 | 78.73 | 0.997 |
| 600 | 0.0267 | 92.50 | 0.998 |
| 800 | 0.0017 | 100.00 | 1.000 |
| 1000-2000 | 0.0007 -> 0.0001 | 100.00 | 1.000 |

Converged at step 800. Validation in JoeyNMT always decodes greedily and dev ==
train here, so that BLEU is the greedy BLEU on the trained pairs.

### Two config traps the overfit test exposed (2026-09-18)
Both were config/data problems, not model bugs. No model code was changed.

1. `data.{src,trg}.max_length: 100` silently dropped 3 of the 50 pairs from
   *training* while leaving the dev set at 50. `bpe200.codes` has only 200
   merges, so it segments very aggressively: the longest of the 50 sentences is
   44 words but 125 subword tokens. Symptom was `num. of seqs: 47` in the epoch
   line, teacher-forced accuracy stuck at 0.843, and a *rising* validation loss
   (the model growing more confident on the 47 it saw and confidently wrong on
   the 3 it never saw). Fixed with `max_length: 512`.
   `scripts/overfit_test.py` now parses `num. of seqs` from `train.log` and fails
   with an explicit message if fewer than N pairs reach the optimizer.

2. The shared `voc_file: test/data/toy/bpe200.txt` (204 entries) cannot represent
   2.5% of the target subwords of these 50 pairs -- it has no entry for `c`, `x`,
   `:` or the typographic apostrophe. 25 of the 50 references contain at least one
   such token. This caps BLEU no matter how good the model is: the run plateaued
   at exactly 71.10 BLEU from step 800 onward while teacher-forced accuracy was
   1.000 and per-token loss was 1.5e-4.
   Fixed by dropping `voc_file` so the vocabulary is built from the 50 pairs.
   BLEU at step 200 went from 21.27 to 81.30 with no model change.
   The script now counts `<unk>` in the hypotheses and names this cause when the
   BLEU assertion fails.

### The 71.10 plateau was NOT a train/inference mismatch (2026-09-18)
Worth recording because it looked like one, and it is the failure mode the
overfit test exists to catch. Teacher-forced accuracy 1.000 with greedy BLEU 71
should be impossible: if argmax is right at every position given the gold prefix,
greedy from BOS must reproduce the reference by induction.
Token-level diagnosis on the checkpoint showed greedy output equal to the gold
token sequence for **all 50 sentences**, the only difference being the trailing
`</s>` that the tokenizer does not emit for the reference. So the full-prefix
decoding path (`transformer_greedy` + `ConvDecoder`) is exactly consistent with
teacher forcing, which is the property SPEC.md §3 relies on. The BLEU gap came
entirely from `<unk>` in the references, i.e. from the vocabulary.

Planned:
- padding-invariance test for the encoder
- gradient-based causality test for the decoder
- overfit-50-sentences test for the full model

### Parameter counts, `scripts/count_params.py` (2026-09-18)
Builds each config through `build_model` and reports totals without loading data
or running a step. Uses the real `voc_file` when it exists and a synthetic
vocabulary otherwise, printing which. Also prints the three-way spread over both
`total` and `body`, which is the number the 10% budget applies to.
Re-run it after `scripts/get_multi30k.sh` to confirm against the real vocabulary;
the `body` column must not move.

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
