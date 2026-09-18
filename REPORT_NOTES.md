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

### Third upstream torch-compatibility fix: ReduceLROnPlateau `verbose` (2026-09-18)
`joeynmt/builders.py` passed `verbose=False` to `ReduceLROnPlateau`. torch
deprecated that argument in 2.2 and later removed it, so Colab's torch 2.11
raises `TypeError` inside `build_scheduler` and training dies before step 1,
while the local torch 2.1.2 still accepts it. Every config we use sets
`scheduling: "plateau"`, so this blocked all three Multi30k runs.

Fix, 6 added lines in the `plateau` branch only:

    if "verbose" not in inspect.signature(ReduceLROnPlateau).parameters:
        kwargs.pop("verbose", None)

Guarded on the signature rather than on a torch version string, so it keeps
forwarding `verbose` wherever it is still accepted and local behaviour is
unchanged (verified: `verbose` is in the signature on torch 2.1.2). No other
scheduler touched. The only visible difference on new torch is that the
`ReduceLROnPlateau(...)` log line no longer prints `verbose=False`, which is
correct: it logs what was actually passed.

This is the third fix of the same kind, all caused by running a January 2024
release (`cdc4d03`, 2024-01-25) in 2026 against 2026 dependency stacks:
1. `numpy<2` -- torch 2.1.2 is built against the NumPy 1.x ABI.
2. `sentencepiece==0.1.99` -- 0.2.x drops `SetVocabulary` from the Python API.
3. this one -- `verbose` removed from `ReduceLROnPlateau`.
The first two are pins, i.e. we pin the world to the code. This one could not
be: torch 2.11 is what Colab provides and downgrading it would give up the GPU
build, so the code had to adapt to the world instead. Worth a sentence in the
report's reproducibility section: pinning stops working once one end of the
stack is not under your control.

### Training schedule sized for a 29k-sentence corpus (2026-09-18)
The three Multi30k configs inherited `epochs: 100`, `updates: 100000`,
`validation_freq: 500` from a WMT-scale template. After the first conv run the
settled values are `epochs: 100` (unchanged), `updates: 30000`,
`validation_freq: 500` (unchanged) and `patience: 5` (unchanged) -- identical in
all three configs, shared block byte-identical, md5 `4590b1c5...` over `data:`
to `model:`. Only `updates` ended up changing, from a decorative 100000 to a
real but non-binding 30000.

Measured, not estimated. Multi30k train is 29,000 pairs; with the real
`bpe.10000.codes` the BPE lengths are mean 13.7 (de) / 13.3 (en), median 13 both
sides, p95 23/22, max 50/45.

The detail that decides the arithmetic is that JoeyNMT's token batching counts
**padded** tokens. `TokenBatchSampler.__iter__` (`joeynmt/datasets.py`) uses
`n_tokens = max(src_len + 1, trg_len + 1)` and closes a batch when
`max_tokens_so_far * len(batch) >= batch_size`, so the longest sentence in a
batch sets the cost for every sentence in it. There is no bucketing. Simulating
that sampler over the real length distribution (5 shuffles, seed 42):

| quantity | value |
|---|---|
| sentences per batch at `batch_size: 4096` | ~121 |
| updates per epoch | ~240 |
| naive `sum(max(src,trg)+1) / 4096` estimate | ~109 |

The naive estimate is 2.2x too low; the gap is padding waste (~45% utilisation).
At the measured ~240 updates/epoch:

| setting | before | after | rationale |
|---|---|---|---|
| `epochs` | 100 = ~24,000 updates | **100**, unchanged | measured at convergence, see below |
| `updates` | 100000 = ~415 epochs | **30000** = ~125 epochs | a real ceiling that still clears the ~24,000 steps 100 epochs takes |
| `validation_freq` | 500 = every 2.08 epochs | **500**, unchanged | already the intended cadence |
| `patience` | 5 = ~10.4 epochs | **5**, unchanged | already the intended tolerance |

`epochs` binds before `updates`, so `updates: 30000` never stops a run; it
replaces an arbitrary large number with a stated ceiling.

### 100 epochs is the right cap: evidence from the first conv run (2026-09-18)
`multi30k_conv.yaml` at commit `d0c7a68`, Colab T4, fp16: 100 epochs / 24,062
steps in 32.4 minutes, dev 32.87 / test 34.16 BLEU with beam 5 (EXPERIMENTS.md).
- Training accuracy was still rising at epoch 100 while dev BLEU had flattened
  near 29.85 greedy, i.e. the model is at convergence on this corpus and further
  epochs would buy overfitting, not quality.
- Early stopping never fired and `ReduceLROnPlateau` never annealed the lr from
  3e-4, so neither `patience` nor `learning_rate_min` shaped this run. The epoch
  cap is the only thing that ended it.
- An intermediate setting of `epochs: 50` / `updates: 20000` was considered and
  dropped: 20,000 updates is ~83 epochs, so it would have truncated training
  before the cap. Hence `updates: 30000`.
- The run also confirms the padded-token measurement below: 24,062 steps over
  100 epochs is 240.6 updates/epoch against a simulated ~240. The naive
  unpadded estimate of ~109 would have predicted ~10,900 steps, less than half
  the truth.
- 32.4 minutes per run means the three-way comparison costs about 1.6 GPU-hours
  in total, so there is no budget reason to shorten training.

On `validation_freq`, an intermediate pass lowered it to 200 on the assumption
of ~110 updates/epoch. Measuring the sampler showed that was the unpadded figure
and the real rate is ~240, at which 500 already gives validation every 2.08
epochs and `patience: 5` gives ~10.4 epochs without improvement -- exactly the
intended behaviour. 200 would have validated every 0.83 epochs, needlessly
often, and would have made `ReduceLROnPlateau` anneal the lr roughly 2.5x more
aggressively than intended (~4.2 epochs of tolerance). It was reverted to 500.
Recorded because the padded-vs-unpadded distinction is easy to get wrong and
changes every schedule number by a factor of 2.2.

`patience` only controls how fast `ReduceLROnPlateau` anneals the lr, not when
training stops: `training.py:735` ends a run at `learning_rate_min` (here 1e-8),
which from 3e-4 with `decrease_factor: 0.7` needs ~29 reductions.

All three models share every one of these settings, along with the data,
tokenizer, vocabulary, beam size 5, length penalty 1.0 and sacrebleu
`tokenize: "13a"`, so the comparison stays controlled: the only difference
between the three runs is the `model:` block.

### Fourth upstream fix: fp16 beam search with the RNN decoder (2026-09-18)
`search.py:411` calls `model.decoder._init_hidden(encoder_hidden)` outside the
`torch.autocast` context. Under fp16 the encoder state arrives as Half while the
bridge layer's weights are still Float, so beam search dies with

    RuntimeError: mat1 and mat2 must have the same dtype, but got Half and Float

Only `RecurrentDecoder` + fp16 + beam search is affected: the Transformer and
conv paths never call `_init_hidden`, and of the three `init_hidden` options only
`"bridge"` does a matmul there (`"last"` slices and repeats, `"zero"` calls
`new_zeros`), so the other two were never at risk.

Fix, 4 added lines in `RecurrentDecoder._init_hidden`, `"bridge"` branch only:

    encoder_final = encoder_final.to(self.bridge_layer.weight.dtype)

`Tensor.to(dtype)` returns the same object when the dtype already matches, so
fp32 is unchanged: no copy, no extra autograd node. Asserted in the tests with
`assertIs`.

Impact on the results table: none, and no retraining. The RNN baseline trained
to completion; only the final beam decode crashed, so the checkpoint was intact
and re-decoding it was enough.

Counted as the fourth upstream fix, but it is a different kind from the first
three and the report should say so:
1. `numpy<2` -- version drift (torch 2.1.2 built against the NumPy 1.x ABI)
2. `sentencepiece==0.1.99` -- version drift (0.2.x drops `SetVocabulary`)
3. `ReduceLROnPlateau` `verbose` -- version drift (removed after torch 2.2)
4. this one -- a plain latent bug in JoeyNMT v2.3, not caused by any dependency
   moving. Verified: it reproduces on the pinned local torch 2.1.2, so no
   version pin would have avoided it. It simply needs fp16 + RNN + beam search
   together, a combination the upstream test suite does not cover.

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

### Scheduler compatibility, `test/unit/test_builders_compat.py` (2026-09-18), 4 tests
The torch 2.11 path cannot be exercised on the local torch 2.1.2, so the test
substitutes a stand-in `ReduceLROnPlateau` whose `__init__` has no `verbose`
parameter and asserts `build_scheduler` still succeeds with the right `mode`,
`factor` and `patience`.
- Teeth, inside the test: it first asserts the stand-in really does raise
  `TypeError` when handed `verbose`. Reverting the shim reproduces exactly the
  Colab failure, `TypeError: __init__() got an unexpected keyword argument
  'verbose'`, and the test errors. Verified by hand.
- `test_verbose_still_passed_where_supported` covers the other direction: on a
  torch that accepts `verbose` the shim must not strip it. The recorder copies
  the real `__signature__`, otherwise patching the class would itself change
  what the shim inspects (this caught a false failure while writing the test).
- `test_other_schedulers_untouched` builds `decaying` and `exponential` and
  checks class and step-at, since the fix must not reach them.

### fp16 `_init_hidden`, `test/unit/test_decoder_fp16.py` (2026-09-18), 4 tests
- `test_bridge_accepts_half_encoder_state`: Half encoder state into a Float
  bridge layer returns finite values of the weights' dtype and the right shape.
- `test_fp32_behaviour_unchanged`: asserts `.to(dtype)` returns the *same object*
  in fp32 (`assertIs`), and that the result equals the hand-computed
  `activation(bridge_layer(x)).unsqueeze(0).repeat(...)`.
- `test_half_and_float_agree`: the Half path matches the Float path to
  rtol/atol 1e-2, i.e. the cast fixes the dtype without changing the answer.
- `test_other_init_hidden_options_accept_half`: `"last"` and `"zero"` also accept
  Half, confirming the fix did not need to reach them.

Teeth: removing the one added line makes two of the four error with exactly the
Colab message, `RuntimeError: mat1 and mat2 must have the same dtype, but got
Half and Float`. Verified by hand, on the local torch 2.1.2 in fp32-only CPU --
which is also the evidence that this bug is not version drift.

## Bugs and fixes

## Environment and reproducibility
- Local: conda env "convs2s", Python 3.11.16, torch 2.1.2, CPU only
- Upstream baseline commit: cdc4d03d430a1b0f29793a0d95743c5e72ae2f6c (2024-01-25)
- Pins required to run JoeyNMT v2.3 in 2026:
  numpy<2 (torch 2.1.2 NumPy 1.x ABI), sentencepiece==0.1.99
  (0.2.x drops SetVocabulary), importlib_metadata (missing from requirements)
- Baseline test suite: OK (skipped=1)
### Colab (GPU training environment), 2026-09-18
- Colab runtime: Python 3.13, torch 2.11.0+cu128. A completely different stack
  from the local Mac (Python 3.11.16, torch 2.1.2, CPU only). Both environments
  are documented because the results come from Colab and the correctness tests
  are run locally.
- `sentencepiece==0.1.99` is required here too (0.2.2 drops `SetVocabulary`).
  No cp313 wheel exists, so pip builds it from source; this succeeds and takes
  under a minute.
- `importlib_metadata` is NOT needed on Python 3.13, unlike the local env.
- JoeyNMT's `setup.py` pins `protobuf<3.21`, which downgrades Colab's
  preinstalled protobuf 5.29.6 and produces dependency-conflict warnings for
  TensorFlow and google-cloud packages. Those packages are unused here and the
  warnings are harmless.
- `torch.cuda.amp.GradScaler` raises a `FutureWarning` on torch 2.11
  (`training.py:112`). Still functional; noted in case it becomes an error.
- Test suite on Colab: 95 tests, `OK (skipped=1)` after the sentencepiece pin.
  (That run predates `test/unit/test_builders_compat.py`; the suite is 99 tests
  locally as of the scheduler fix below.)

## Results and observations

### Final results, Multi30k de-en, test set, beam 5 (2026-09-18)

| model | params | test BLEU | dev BLEU | throughput | wall-clock |
|---|---|---|---|---|---|
| ConvS2S (4 enc / 3 dec, k=3) | 10,983,936 | **34.16** | 32.87 | ~30k tok/s | 32.4 min |
| Transformer (3 enc / 2 dec) | 10,862,336 | **35.06** | 34.28 | ~26k tok/s | 33.9 min |
| biGRU + Luong (2 enc / 2 dec) | 11,116,032 | **16.66** | 16.02 | ~10.2k tok/s | 72.7 min |

Parameter counts within 2.3% of each other, and data, tokenizer, vocabulary,
beam size 5, length penalty 1.0, sacrebleu `tokenize: "13a"` and the training
schedule are identical across the three (EXPERIMENTS.md, and the shared config
block md5 check).

Observations:
- The conv model is 0.90 BLEU below the Transformer at matched parameters, the
  direction and rough magnitude reported in the literature.
- Throughput on the same T4: conv ~30k tok/s, transformer ~26k, RNN ~10.2k. The
  conv model is ~3x the RNN and slightly ahead of the Transformer, which is the
  parallelism argument in Gehring et al. section 1 -- convolutions over the
  whole sequence instead of a sequential recurrence. Note this is *training*
  throughput; we gave up ConvS2S's inference-speed advantage by decoding the
  full prefix each step (see the design decision on incremental state).

LIMITATION -- must appear in the report, not buried:
- All three ran a fixed 100-epoch budget. Final training accuracy was 0.581
  (conv), 0.630 (transformer), 0.449 (RNN), all still rising. None had fully
  converged, and the RNN least of all, so its 16.66 understates what the
  architecture can do.
- An equal-epoch budget systematically disadvantages the slowest-converging
  architecture. The RNN also had the lowest throughput, so equal epochs cost it
  2.2x the wall-clock of the other two and still left it furthest from
  convergence. An equal-wall-clock or train-to-convergence protocol would be the
  fairer comparison; equal-epoch was chosen for simplicity and should be named
  as a threat to validity rather than defended.
- Single seed per model, so there is no variance estimate. Differences under
  ~1 BLEU should not be treated as significant.
- Those last two points interact and the report must not dodge it: the
  conv-to-transformer gap is 0.90 BLEU, which is *below* our own significance
  threshold. The honest claim is that ConvS2S and the Transformer are
  indistinguishable at this scale and budget, not that the Transformer wins.
  The RNN's ~18 BLEU deficit is far outside that band and is a real effect, but
  is confounded by the convergence limitation above.


## Limitations and open questions

## References
- Gehring et al. 2017, Convolutional Sequence to Sequence Learning
- Dauphin et al. 2017, Language Modeling with Gated Convolutional Networks
- Kreutzer et al. 2019, Joey NMT: A Minimalist NMT Toolkit for Novices
