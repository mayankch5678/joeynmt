# Report outline

Every bullet is sourced from REPORT_NOTES.md (RN), EXPERIMENTS.md (EX), SPEC.md
(SP) or the configs. Items marked **[GAP]** are needed by the section but are not
in the notes yet; they are collected again at the end.

---

## 1. Introduction

- Task: implement Gehring et al. (2017) ConvS2S as a new encoder/decoder pair in
  JoeyNMT v2.3 (upstream commit `cdc4d03`, 2024-01-25) and compare it with
  JoeyNMT's own RNN and Transformer at matched parameter count on Multi30k de-en.
- Design rule: ADD, never REPLACE. `encoder.type: conv` / `decoder.type: conv`
  are selected by config; RNN and Transformer code paths stay runnable.
- Headline result: ConvS2S 34.16, Transformer 35.06, biGRU+Luong 16.66 test BLEU
  (beam 5). The 0.90 gap between ConvS2S and the Transformer is below our own
  ~1 BLEU significance threshold; the RNN's ~18 BLEU deficit is far outside it
  but confounded by under-training (RN, Results).
- Secondary finding: training throughput on one T4 is ConvS2S ~30k tok/s,
  Transformer ~26k, RNN ~10.2k.
- Contribution list to state: (a) the ConvS2S implementation, (b) four
  property-based correctness tests plus the overfit gate, (c) a
  parameter-matched three-way comparison, (d) four upstream JoeyNMT
  incompatibilities found and fixed while doing it.
- **[GAP]** Motivation and one paragraph of related work (RN cites only three
  references and has no prose on why this comparison matters).

## 2. Background: ConvS2S architecture (SP)

- **GLU** (SP §1): conv maps C -> 2C, split `(A, G)`, output `A ⊙ σ(G)`.
- **Encoder** (SP §2.1, E1-E15): word + learned positional embedding, dropout,
  linear E -> H; `L_e` residual GLU blocks `z^l = √0.5 (y + z^{l-1})` with
  symmetric padding `(k-1)/2` (length preserved: `S + (k-1) - k + 1 = S`), pads
  masked to zero before every conv; linear H -> E gives keys `z^u`; values
  `z^c = √0.5 (z^u + e)`.
- **Decoder** (SP §3.1, D1-D19): left-pad `k-1` so position i sees only
  `i-k+1 … i`; per layer, a GLU then attention with query `d = W h̃ + b + g`
  (g = target embedding), `a = softmax(d z^uᵀ)` masked over pads, context
  `c = a z^c`, rescaled by `m_j · √(1/m_j)`, then `o = √0.5 (h̃ + c_h)` and
  `h = √0.5 (o + h_prev)`.
- **Multi-step attention**: every one of the `L_d` decoder layers attends
  (SP §3.1). The `+ e_j` in the values is what distinguishes it from attending
  over `z` alone.
- **Initialisation** (SP §5): `N(0, √(4p/n))` into GLU convs, `N(0, √(1/n))`
  into other layers, `N(0, 0.1)` embeddings and positional tables, biases 0,
  pad rows 0; `p = 1 - dropout`, `n = H·k` for convs.
- **Gradient scaling** (SP E15): encoder gradients scaled by `1/L_d`.
- **Where each √0.5 applies** (SP §4): four places (E9, E12, D15, D16).
- **Weight normalisation** on the GLU convs (Salimans & Kingma, via the paper).
- Figure candidates: one encoder block, one decoder block with its two nested
  residuals (D15 inside D16). **[GAP]** no figures exist yet.
- **[GAP]** Reference numbers from Gehring et al. (their WMT/IWSLT BLEU, model
  sizes) to position our small-model results. RN says the 0.90 gap is "the
  direction and rough magnitude reported in the literature" without citing a
  number.

## 3. Implementation

### 3.1 JoeyNMT integration (all edits to existing files)

- New code: `ConvEncoder`, `ConvDecoder` (added to `encoders.py` /
  `decoders.py`; 367 insertions with one changed line, the `List` import),
  `GLUConvBlock` and the multi-step attention module in `conv_layers.py`.
- `model.py` (5 lines replaced): an `elif type == "conv"` before the recurrent
  fallback in both the encoder and decoder branches, plus imports. Order
  matters: the existing code was if/else, so without the `elif` `type: conv`
  silently built an RNN.
- `model.py`: `encoder.num_attention_layers = len(decoder.layers)`, guarded by
  `isinstance`, is what switches on encoder gradient scaling (E15).
- `model.py`: `reset_parameters()` re-run on both conv modules right after
  `initialize_model`, with `init_output_layer=not tied` (see 3.2).
- `search.py`: import plus `(TransformerDecoder, ConvDecoder)` in the `greedy`
  dispatch (line 45) and in `is_transformer` (line 398). The `RecurrentDecoder`
  branch is untouched, so RNN and Transformer decoding are bit-identical.
- Three further edits to existing files, all forced by running the 2024 release
  in 2026 (Section 5 / Appendix): `builders.py` (`verbose` shim, 6 lines,
  `plateau` branch only), `RecurrentDecoder._init_hidden` (4 lines, `"bridge"`
  branch only), and the `initialization.py` NameError, which is documented but
  not fixed.
- Honest statement: the "existing code untouched" rule holds for behaviour on
  the original paths, but two baseline files (`builders.py`, `decoders.py`
  `RecurrentDecoder`) did receive compatibility edits. Say so.

### 3.2 Design decisions and reasoning

- **No incremental decoding state.** JoeyNMT's decoders re-run the full prefix
  each step (`search.py:160`, `508-532`), so the same left-pad + trim code is
  correct in training and inference. Cost O(T²) per sentence, identical to the
  Transformer baseline, so the comparison is fair. Consequence: we explicitly
  give up ConvS2S's inference-speed advantage.
- **Causality from padding only.** `transformer_greedy` passes a dummy
  `trg_mask = new_ones([1,1,1])`, so `trg_mask` cannot supply causality; it
  comes from left-pad(k-1) + trim(k-1). This is why the causality test is
  gradient-based.
- **Encoder returns keys and values concatenated**, `(B, S, 2E)`. The
  `encoder_hidden` slot is hardcoded `None` on the Transformer decode path
  (`search.py:242, 518, 543`); `encoder_output` is the only tensor beam search
  tiles and re-indexes. Stashing `z^c` on the module would survive greedy but
  not beam reordering. `output_size` stays `H` (Noam scheduler, `training.py:121`).
- **Weight norm via `torch.nn.utils.parametrizations.weight_norm`**; the legacy
  one is deprecated in torch 2.1.2. Re-init by assigning `module.weight` under
  `no_grad`; `right_inverse` recomputes `g = ||v||`, so parameter identity is
  stable.
- **Weight norm on convs only**, not on the six `Linear` layers (fairseq does
  both). Deviation, revisit if unstable; candidate ablation.
- **Dropout on the conv input**, not after the GLU: matches the paper and
  fairseq, and is what `p` in `√(4p/n)` assumes.
- **Projections conditional**: `input_proj`/`output_proj` exist only when
  `E != H`.
- **`GLUConvBlock(residual: bool = True)`**: the decoder's attention sits between
  the GLU and the block residual (D15 nested in D16), so the decoder passes
  `residual=False` and closes the residual itself. Encoder default unchanged.
- **Attention rescale is `√m` with `m` the real source length**
  (`src_mask.sum(-1)`), not the padded length; the padded length would make the
  output depend on the rest of the batch.
- **Per-layer attention stored detached** (`layer_attentions`) to avoid keeping
  the autograd graph alive; rebuilt every forward.
- **Softmax tying** works for the conv decoder without the `E == H` constraint:
  `output_layer` is `Linear(E, V, bias=False)`, `build_model` injects
  `emb_size`; verified `weight is trg_embed.lut.weight`, both `(204, 64)`.
  The final Multi30k configs do not tie.
- **Init ordering bug (fixed).** `initialize_model` ran after the conv
  constructors and overwrote ConvS2S init. Measured with H=64, k=3, dropout 0.1
  (n=192, p=0.9, target std 0.13693): GLU conv weight std **0.00937** before
  the fix (~15x too small; not xavier's 0.0589 either, because xavier
  initialised the weight-norm's `v` and `g` tensors separately, and
  `fan_in=1` on the `(2H,1,1)` `g` tensor mis-scales `g·v/||v||`), **0.13693**
  after. The `tied_softmax` guard exists because in-place re-init would
  overwrite the target embedding table and un-zero its pad row.
- **Paper ambiguities and the chosen reading** (SP §6):
  A. `d = W h̃ + b + g` unscaled (fairseq uses `√0.5 (…)`);
  B. non-GLU init `N(0, √(1/n))` (paper's dropout-aware form is `√(p/n)`);
  C. `√0.5` applied to `z^u + e`;
  D. learned positional tables inside the conv modules, since JoeyNMT's
     `Embeddings` has none;
  E. returned `att` is the last layer's, matching `TransformerDecoder`.
- **Pre-existing upstream bug (not ours)**: `initialization.py:136-139` defines
  `deepnet` only for transformer/transformer, but `:194` reads it for
  `xavier_normal`, so any non-transformer model with `xavier_normal` raises
  `NameError`. Configs use `xavier_uniform`.

### 3.3 Tooling

- `scripts/count_params.py`: builds each config through `build_model`, no data.
  Bug in its first version: substring match on `"output_layer"` also caught
  `MultiHeadedAttention.output_layer`, understating the Transformer body by
  592,128 and making a 3+3/ff512 Transformer look +2.3% when it is +20.4%.
  Now matches the three exact parameter names.
- `scripts/overfit_test.py`, `scripts/get_multi30k.sh`.
- **[GAP]** Lines-of-code / file inventory table for the implementation.

## 4. Correctness evidence

Selection note: the notes contain ~20 conv tests; I took the four that each
prove a distinct property the architecture can get wrong. **[GAP]** Confirm
this is the intended four (see flags).

### 4.1 Encoder padding invariance (`test_conv_encoder_padding_invariance`)
- Proves: a sentence encoded alone equals the same sentence inside a padded
  batch at its real positions (`assert_close`, atol 1e-6) with random garbage in
  the pad slots; the padded tail is exactly zero. No pad position leaks into a
  real one.
- Verified able to fail: disabling the per-block mask leaves sentence 0 (longest,
  unpadded) unchanged but breaks sentences 1 and 2 by **0.58 and 0.61** max abs
  diff, ~5 orders of magnitude above tolerance.

### 4.2 Causality, block and decoder (`test_glu_conv_block_causality`, `test_conv_decoder_causality`)
- Proves: for k in {3,5}, N=T=8, backprop from `output[0, i, :].sum()`; gradient
  at every j > i is **exactly 0.0**. Decoder version backprops into `trg_embed`.
  Causality comes from left-pad + trim alone, never from `trg_mask`.
- Verified able to fail: (i) `test_glu_conv_block_causality_test_has_teeth` reruns
  the identical check with `padding_mode="symmetric"` and requires strictly
  positive future gradient at every position but the last; (ii) the decoder test
  also asserts the j = i gradient is strictly positive, so zeros cannot come
  from a dead gradient path.
- Companions: `test_conv_decoder_dummy_trg_mask` (the `[1,1,1]` placeholder
  gives bit-identical logits to a real mask), attention rows sum to 1 with
  exactly zero mass on padded source positions (source lengths [6,4,1]).

### 4.3 ConvS2S init survives `build_model` (`test_conv_model_init.py`, 5 tests)
- Proves: every GLU conv weight has std within 5% of `√(4p/n)` and at least 30%
  away from xavier's value; non-GLU projections match `√(1/n)`; biases exactly
  0; with tying, `output_layer.weight is trg_embed.lut.weight` and the pad row
  is still exactly 0.
- Verified able to fail: the test first asserts the two hypotheses are >50%
  apart, so it cannot pass by them being indistinguishable. Mutation: disabling
  the two `reset_parameters()` calls in `build_model` fails **3 of 5** (GLU std
  0.00937 vs 0.13693; projection 0.14470 vs 0.17678; untied output layer off
  by 48%). Tying guard: calling `reset_parameters(init_output_layer=True)` by
  hand does destroy the pad row.

### 4.4 Overfit-50 gate (`scripts/overfit_test.py`, `configs/conv_overfit.yaml`)
- Proves: the whole stack (dispatch, forward/backward, greedy via
  `transformer_greedy`, tokenizer, BLEU) can memorise. 50 toy pairs as both
  train and dev, H=64, 2 layers, k=3, no dropout, Adam 1e-3, CPU. PASS: per-token
  loss **0.0001** (target < 0.1), greedy BLEU **100.00** (target > 95),
  teacher-forced accuracy **1.000**, converged at step 800
  (loss 0.1277 / BLEU 81.30 at 200; 0.0017 / 100.00 at 800).
- Verified able to fail: it did fail, twice, for reasons outside the model.
  (1) `max_length: 100` silently dropped 3 of 50 pairs from training (44 words
  = 125 BPE tokens under 200 merges): accuracy stuck at 0.843, dev loss rising.
  (2) `bpe200.txt` (204 entries) has no entry for `c`, `x`, `:` and the
  typographic apostrophe; 25 of 50 references had one, BLEU plateaued at exactly
  **71.10** with accuracy 1.000. The script now checks `num. of seqs` and counts
  `<unk>` in hypotheses and names the cause.
- Key finding: the 71.10 plateau was not a train/inference mismatch. Token-level
  diagnosis showed greedy output equals the gold sequence for all 50
  sentences, apart from the trailing `</s>`. That confirms the full-prefix
  decoding path is exactly consistent with teacher forcing (SP §3).
- **[GAP]** No mutation check for this test on the *model* side (e.g. a
  deliberately leaky decoder). Its failure evidence is data/config only.

### 4.5 Supporting evidence (one paragraph)
- Encoder: output shape over k in {1,3,5} and S in {1,2,5,11}; closed-form
  parameter count; every leaf has a finite non-zero grad; grad reaching
  `src_embed` is exactly 1/4 of unscaled with `num_attention_layers=4`; freeze.
- Decoder: shapes over k and T in {1,2,5,9}; per-layer attention for L_d in
  {1,3,6}; freeze.
- Compatibility fixes: `test_builders_compat.py` (4 tests; the stand-in raises
  `TypeError: __init__() got an unexpected keyword argument 'verbose'`, and
  reverting the shim reproduces it), `test_decoder_fp16.py` (4 tests; removing
  the one added line gives `RuntimeError: mat1 and mat2 must have the same
  dtype, but got Half and Float`; `assertIs` proves fp32 is unchanged).
- Suite growth: 77 (baseline) -> 84 -> 90 -> 95 -> 99 -> **103, `OK (skipped=1)`**
  (103 verified locally in the `convs2s` env for this outline). Colab ran 95.
- Smoke run on `conv_small.yaml`: 32 updates, 767,424 params, loss 2946.91,
  greedy and beam both run.
- **[GAP]** No cross-check against a reference implementation (fairseq fconv).
  All evidence is property-based; nothing shows our numbers match an
  independent implementation. Worth stating as a limitation or closing with
  one numerical comparison.

## 5. Experimental setup

### 5.1 Data and preprocessing
- Multi30k de-en: train 29,000, dev 1,014, test 1,000 pairs.
- BPE via subword-nmt 0.3.8, 10,000 merges, `bpe.10000.codes` learned on the
  **training split only**; `lowercase: False`, `normalize: False`,
  `pretokenizer: "none"`; shared source/target `vocab.txt` (10,020 lines + 4
  specials = V = **10,024**).
- Measured BPE lengths: mean 13.7 (de) / 13.3 (en), median 13 both, p95 23/22,
  max 50/45. `max_length: 100`.
- Evaluation: sacrebleu 2.6.0 (local), `tokenize: "13a"`, beam 5, length penalty
  1.0, `max_output_length: 100`.

### 5.2 Matched parameters
- Identical `data:` through `model:` blocks in all three configs (same md5,
  `4590b1c5…` at the final commit); only `model:` differs.
- Vocabulary-independent sizing: every model uses `embedding_dim: 256`, ties
  nothing, ends in a width-256 output layer, so each has exactly `3·V·256`
  vocabulary parameters. Checked at V=10000 and V=6000: `body` identical.
  RNN decoder hidden pinned to 256 for this reason.
- Sizes: conv H=256, 4 enc / 3 dec, k=3, dropout 0.2; Transformer H=256, ff 512,
  8 heads, 3 enc / 2 dec, pre-LN, ReLU (a 3/3 split at ff 512 is +20.4%, off
  budget); RNN 2-layer biGRU encoder (output 512) + 2-layer GRU decoder, Luong
  attention, input feeding, bridge init.
- Counts at V=10,000 (`count_params.py`):

  | config | total | shared | body | encoder | decoder |
  |---|---|---|---|---|---|
  | conv | 10,965,504 | 7,680,000 | 3,285,504 | 1,642,496 | 1,643,008 |
  | transformer | 10,843,904 | 7,680,000 | 3,163,904 | 1,581,824 | 1,582,080 |
  | rnn | 11,097,600 | 7,680,000 | 3,417,600 | 1,972,224 | 1,445,376 |

  At the real V=10,024 the totals are 10,983,936 / 10,862,336 / 11,116,032 (each
  +18,432 = 3·24·256). Spread **8.0%** on body, **2.3%** on total.
- Alternatives inside the budget (recorded, not run): Transformer ff 384 /
  4 enc / 2 dec (+0.4% vs conv); RNN hidden 384 / 1 enc / 2 dec (5.3% spread).

### 5.3 Optimisation and schedule (shared)
- Adam, betas (0.9, 0.998), lr 3e-4, plateau scheduler (patience 5, factor
  0.7, min 1e-8), grad clip 1.0, label smoothing 0.1, token batches of 4096,
  fp16, `random_seed: 42`, `validation_freq: 500`, `epochs: 100`,
  `updates: 30000` (non-binding), early-stopping metric BLEU, `keep_best_ckpts: 3`.
- Why these numbers: JoeyNMT batches by **padded** tokens
  (`n_tokens = max(src+1, trg+1)`, closes when `max·len >= 4096`, no
  bucketing), so ~121 sentences per batch and ~**240 updates/epoch**, 2.2x the
  naive unpadded estimate of ~109 (~45% utilisation). Confirmed by the run:
  24,062 steps / 100 epochs = 240.6.
- `epochs` binds, `updates` never does. `patience` only anneals lr; training
  ends at `learning_rate_min`, which needs ~29 reductions from 3e-4 at 0.7.
- A first pass lowered `validation_freq` to 200 on the unpadded estimate; it was
  reverted to 500 after measuring (200 would validate every 0.83 epochs and
  anneal ~2.5x too aggressively).

### 5.4 Two environments
- **Local**: MacBook Air, CPU only, conda `convs2s`, Python 3.11.16, torch 2.1.2,
  numpy 1.26.4, sentencepiece 0.1.99, sacrebleu 2.6.0, `importlib_metadata`
  added. Used for correctness only (tests, overfit, smoke).
- **Colab**: T4, Python 3.13, torch 2.11.0+cu128, sentencepiece 0.1.99 built
  from source (no cp313 wheel; <1 min), `importlib_metadata` not needed,
  `protobuf<3.21` from JoeyNMT's `setup.py` downgrades Colab's 5.29.6 (harmless
  TF/GCP conflict warnings), `torch.cuda.amp.GradScaler` FutureWarning
  (`training.py:112`). Every number in the results table comes from here.
- Rule: anything that produces a table number runs on GPU; anything that checks
  correctness runs locally. Consequence to state: the correctness tests were
  run on torch 2.1.2, the results on torch 2.11 (Colab ran the suite once at
  95 tests, OK).
- **[GAP]** Colab package list is not frozen (only `requirements-frozen.txt`
  for local). Colab commit/notebook, GPU driver, exact sacrebleu version there.

## 6. Results

### 6.1 Main table (test set, beam 5; EX)

| model | params | test BLEU | dev BLEU (beam 5) | best greedy dev | throughput | wall-clock |
|---|---|---|---|---|---|---|
| ConvS2S 4/3, k=3 | 10,983,936 | **34.16** | 32.87 | 29.85 (step 23000) | ~30k tok/s | 32.4 min |
| Transformer 3/2 | 10,862,336 | **35.06** | 34.28 | 31.87 (step 24000) | ~26k tok/s | 33.9 min |
| biGRU + Luong 2/2 | 11,116,032 | **16.66** | 16.02 | 14.49 (step 24000) | ~10.2k tok/s | 72.7 min |

All: 24,062 steps / 100 epochs, Colab T4 fp16, seed 42, single run.

### 6.2 Throughput
- Conv ~30k > Transformer ~26k > RNN ~10.2k tok/s: conv ≈ 3x RNN, ≈ 1.15x
  Transformer. Matches the parallelism argument in Gehring et al. §1.
- This is **training** throughput. Decoding was full-prefix for all three, so
  no inference-speed advantage is claimed or measured.
- Equal epochs cost the RNN 2.2x the wall-clock of the other two.
- Total cost of the comparison ~1.6 GPU-hours (+ RNN re-decode).

### 6.3 Honest reading (these must be in the text, not the appendix)
- ConvS2S vs Transformer: 0.90 BLEU, below the ~1 BLEU single-seed threshold.
  Claim: indistinguishable at this scale and budget, **not** "Transformer wins".
- RNN: ~18 BLEU behind, a real effect, but confounded (below).
- Fixed 100-epoch budget. Final training accuracy 0.581 (conv), 0.630
  (Transformer), 0.449 (RNN), all still rising. Equal-epoch penalises the
  slowest converger; the RNN's 16.66 understates the architecture. Equal
  wall-clock or train-to-convergence would be fairer; equal-epoch is a
  threat to validity, not a defended choice.
- Single seed: no variance estimate.
- ConvS2S evidence on the epoch cap: dev greedy BLEU flattened near 29.85, early
  stopping never fired, lr never annealed from 3e-4. Transformer dev BLEU was
  still rising at the last validation. **These two statements sit uneasily
  with the limitation above; see flags.**
- Process incidents worth a sentence each: RNN beam decode crashed on the fp16
  `_init_hidden` bug and was re-decoded from the intact checkpoint (no
  retraining); commit hashes for the Transformer and RNN rows are inferred.
- **[GAP]** Training curves (dev BLEU vs step, three models on one plot).
  `scripts/plot_validations.py` exists but no figure is recorded.
- **[GAP]** Attention visualisation from `layer_attentions` (built for exactly
  this, no figure yet).
- **[GAP]** Example translations / error analysis; BLEU signature string;
  significance test (bootstrap resampling on the test set would give a CI
  without new training).

## 7. Ablations

Placeholder, not yet run. Nothing in EXPERIMENTS.md. Candidates already
motivated by the notes:

- Weight norm on the six `Linear` layers as well (deviation from fairseq, RN §3.2).
- Attention query `√0.5 (W h̃ + b + g)` (fairseq) vs paper form (SP §6 A).
- Multi-step attention vs last layer only (the core ConvS2S claim; needs a
  config switch, not present yet).
- ConvS2S init on vs off: the 0.00937 vs 0.13693 std measurement exists, the
  BLEU effect does not. Cheapest to run because the "off" variant is the
  pre-fix generic init.
- Encoder gradient scaling on vs off (`num_attention_layers=None`).
- Kernel width k, depth.
- Equal-wall-clock or train-to-convergence rerun of the three-way comparison
  (the largest threat to validity, arguably the most valuable "ablation").
- Multiple seeds.
- Each row will need an EXPERIMENTS.md line first.

## 8. Limitations

- Equal-epoch budget, all three still improving (6.3).
- Single seed; <1 BLEU differences not significant.
- No incremental decoding: ConvS2S's inference advantage given up by design;
  O(T²) for all three.
- Small scale: 29k pairs, ~11M params, a 4/3 layer conv net versus the paper's
  8-15 layer models; conclusions may not transfer to WMT scale.
- Weight norm on convs only; fairseq applies it to Linear as well.
- Positional tables capped at `max_position: 256`.
- No reference-implementation numerical cross-check.
- Validation is always greedy in JoeyNMT while the reported dev/test is beam 5,
  so "best dev BLEU" (checkpoint selection) and reported dev BLEU differ.
- Two environments with different torch (2.1.2 vs 2.11); tests do not run on the
  training stack.
- Pinning stops working when one end of the stack is not under your control:
  three of the four upstream fixes were version drift (`numpy<2`,
  `sentencepiece==0.1.99` were pinnable; `ReduceLROnPlateau verbose` removed after
  torch 2.2 could not be, because downgrading Colab's torch loses the GPU
  build). The fourth (fp16 `_init_hidden`) is a plain latent JoeyNMT bug that
  reproduces on torch 2.1.2, so no pin would have avoided it.
- Baseline commit hashes for two of three runs are inferred.

## 9. Conclusion

- ConvS2S was added to JoeyNMT without changing baseline behaviour, and passes
  causality, padding-invariance, init and overfit checks, each shown able to fail.
- At ~11M parameters on Multi30k, ConvS2S (34.16) and the Transformer (35.06)
  are indistinguishable within single-seed noise; both far ahead of the
  biGRU (16.66), whose deficit is partly a budget artefact.
- Conv trains fastest per token (~30k tok/s).
- Future work: ablations (Section 7), equal-wall-clock protocol, multiple
  seeds, incremental decoding to test the inference-speed claim.

## Appendix: reproducibility

### A. Configs
- `configs/multi30k_{conv,transformer,rnn}.yaml` (shared block md5 `4590b1c5…`;
  check with `sed -n '/^data:/,/^model:/p' configs/multi30k_*.yaml | md5`).
- `configs/conv_small.yaml` (toy smoke), `configs/conv_overfit.yaml` (gate).
- Model blocks as in 5.2; full text is in the repo.

### B. Pins and environments
- Local: Python 3.11.16, torch 2.1.2, numpy 1.26.4, sentencepiece 0.1.99,
  sacrebleu 2.6.0, subword-nmt 0.3.8, `importlib_metadata`; full list in
  `requirements-frozen.txt`.
- Colab: Python 3.13, torch 2.11.0+cu128, sentencepiece 0.1.99 (source build).

### C. Seeds
- `random_seed: 42` in all three configs. Sampler simulation also used 42.
  One seed per model.

### D. Commits

| item | commit |
|---|---|
| upstream JoeyNMT v2.3.0 | `cdc4d03d430a1b0f29793a0d95743c5e72ae2f6c` (2024-01-25) |
| env setup / pins | `2eefbcd` |
| conv wired into build_model + search | `089e78b`, `fcafb19` |
| overfit-50 gate | `eb52a09` |
| Multi30k data prep + configs + param counting | `35422ed` |
| scheduler `verbose` fix | `d0c7a68` (conv run trained here) |
| schedule finalised | `550e2ca` (Transformer, RNN training) |
| fp16 `_init_hidden` fix | `b5f54a9` (RNN re-decode) |
| three-way results logged | `3a937c5` |

Note: `d0c7a68` and `550e2ca` differ in the config (see flags).

### E. Reproduction commands
Derived from JoeyNMT's CLI and the repo scripts, **not** recorded from the
Colab notebook (see flags):

    conda activate convs2s
    pip install -e . && pip install "numpy<2" sentencepiece==0.1.99 importlib_metadata
    python -m unittest                                  # OK (skipped=1), 103 tests
    python scripts/overfit_test.py --config configs/conv_overfit.yaml -n 50
    python scripts/count_params.py configs/multi30k_*.yaml
    bash scripts/get_multi30k.sh                        # data + BPE + vocab
    python -m joeynmt train configs/multi30k_conv.yaml  # on Colab T4
    python -m joeynmt test  configs/multi30k_conv.yaml

- Upstream fixes list: `numpy<2`, `sentencepiece==0.1.99`, `verbose` shim in
  `builders.py`, fp16 `_init_hidden`; plus the un-fixed `xavier_normal` NameError.

---

# Flags: what the sections need that the notes don't have

**Needs your decision**
1. **"The four tests."** RN lists ~20 conv tests and only three "Planned"
   headline checks (padding invariance, causality, overfit-50). I used
   padding invariance, causality, init-after-build_model, overfit-50. If you meant
   a different four (e.g. counting encoder and decoder causality separately),
   Section 4 needs regrouping.

**Inconsistencies in the notes to reconcile before writing**
2. **Conv config differs from the other two runs.** The conv row is commit
   `d0c7a68`, whose config still has `updates: 100000` and no patience comment;
   `550e2ca` changed those in all three. The "identical shared block, md5
   `4590b1c5…`" claim holds for the final files, not for the file the conv run
   used. Harmless (`updates` never binds at 24,062 steps) but the report should say
   it, not claim byte-identical.
3. **"Converged" vs "none converged."** RN says the conv run was "at
   convergence" (greedy dev BLEU flat near 29.85, more epochs would buy
   overfitting) and, later, that all three were "still rising" and none had
   converged. Note also that conv's best greedy BLEU was at step 23000 of 24000
   while the Transformer's was at 24000. Pick one story; the second is the
   more cautious one.
4. **Stale text.** RN still has a "Planned:" list of three tests that now exist,
   an empty "Bugs and fixes" section, an empty "Limitations and open questions"
   section, and a Colab test count of 95 / "99 locally" where the suite is now
   103. This outline pulls the content from where it actually lives.
5. **Commit hashes** for the Transformer and RNN rows are inferred (EX says so).
   The Colab notebooks are the only source of truth.

**Missing from the notes entirely**
6. Reference numbers from Gehring et al. (and the literature the "direction and
   magnitude" claim relies on): needed for Background and Results.
7. Colab: frozen package list, notebook/commit, exact sacrebleu version and
   BLEU signature, the actual train/test commands.
8. How throughput was measured (which tokens, which log line, averaged over
   what). It is a headline finding and currently "~".
9. Training curves, attention plots, example translations: none saved.
10. Any statistical test. A bootstrap over the 1,000 test sentences would put a
    CI on the 0.90 gap with no retraining.
11. Ablation results (Section 7) and the ablation config switches
    (multi-step vs last-layer attention has none).
12. Independent numerical check against fairseq's fconv.
13. A model-side mutation check for the overfit test (leaky decoder), to match
    the evidence the other three tests have.
14. Which Multi30k test split (test2016 Flickr vs others); Section 5.1 should
    name it.
15. Motivation, related work, and a file/LOC inventory for Introduction and
    Implementation.
16. A note that the `RecurrentDecoder` and `builders.py` edits are exceptions to
    "existing code untouched" (behaviour-preserving, but the rule as written in
    CLAUDE.md is stricter).
