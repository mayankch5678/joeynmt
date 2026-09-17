# CODEBASE_MAP.md

Map of JoeyNMT v2.3.0 (baseline commit `cdc4d03`), written on branch `convseq2seq`
at `c1dcb02` for adding a ConvS2S encoder/decoder. All line numbers refer to that
commit.

Notation: `B` = batch size, `S` = source length, `T` = target length,
`H` = hidden size, `E` = embedding dim, `V` = target vocab size, `k` = beam size.
Floats are `torch.float32` (or `float16` inside `torch.autocast` on CUDA when fp16
is on). Masks are `torch.bool`. Token ids are `torch.long`.

---

## 1. `configs/` and `scripts/`

### configs/
| File | One line |
|---|---|
| `iwslt14_deen_bpe.yaml` | Transformer, IWSLT14 de→en, plain-text data with subword-nmt BPE (needs `scripts/get_iwslt14_bpe.sh`). |
| `iwslt14_deen_sp.yaml` | Transformer, IWSLT14 de→en, loaded through HuggingFace datasets, SentencePiece. |
| `jparacrawl_enja_sp.yaml` | Transformer, JParaCrawl en→ja, HuggingFace, SentencePiece. |
| `jparacrawl_jaen_sp.yaml` | Transformer, JParaCrawl ja→en, HuggingFace, SentencePiece. |
| `rnn_reverse.yaml` | Recurrent enc/dec on the synthetic sequence-reversal task (short sanity run). |
| `rnn_small.yaml` | Recurrent enc/dec on `test/data/toy`. Fully commented reference config listing every option. |
| `transformer_reverse.yaml` | Transformer on the synthetic reversal task. |
| `transformer_small.yaml` | Transformer on `test/data/toy`. Fully commented reference config for the Transformer. |
| `wmt17_ende_bpe.yaml` | Transformer, WMT17 en→de, HuggingFace, BPE. |
| `wmt17_ende_sp.yaml` | Transformer, WMT17 en→de, HuggingFace, SentencePiece (header still says `joeynmt_version: "2.0.0"`). |

None of these is a Multi30k config. The Multi30k de-en baselines still have to be
written as new config files.

### scripts/
| File | One line |
|---|---|
| `average_checkpoints.py` | Averages the parameters of several checkpoints (fairseq-style). |
| `build_vocab.py` | Builds vocab files (and optionally trains SentencePiece/BPE) from a config's training data. |
| `discord_joey.py` | Discord bot that serves translations from trained JoeyNMT models. |
| `generate_reverse_task.py` | Generates the synthetic sequence-reversal dataset (random digit sequences → reversed). |
| `get_iwslt14_bpe.sh` | Downloads and preprocesses IWSLT14 de-en with Moses tokenization and BPE into `test/data/iwslt14`. |
| `get_iwslt15_envi.sh` | Downloads the Stanford IWSLT15 en-vi preprocessed data into `test/data/iwslt_envi`. |
| `plot_validations.py` | Plots validation curves from one or more models' `validations.txt`. |

---

## 2. Encoder / Decoder base classes

### Base classes: no `__init__`, no `forward`

`joeynmt/encoders.py:15-28`
```python
class Encoder(nn.Module):
    # pylint: disable=abstract-method
    @property
    def output_size(self):
        return self._output_size
```
`joeynmt/decoders.py:18-32`
```python
class Decoder(nn.Module):
    # pylint: disable=abstract-method
    @property
    def output_size(self):
        """Return the output size (size of the target vocabulary)"""
        return self._output_size
```
The only rule the base classes set is **the subclass must set `self._output_size`**:
- Encoder: the size of its output states (`H`). `RecurrentDecoder` reads it through
  `encoder.output_size`, and `TrainManager` reads `model.encoder._output_size` for
  the Noam scheduler (`training.py:121`).
- Decoder: `V`. `beam_search` reads it through `model.decoder.output_size`
  (`search.py:394`).

The input/output contract is set by how `Model` and `search.py` call the
subclasses, as shown below.

### `TransformerEncoder` (`encoders.py:168`)
```python
def __init__(self, hidden_size: int = 512, ff_size: int = 2048, num_layers: int = 8,
             num_heads: int = 4, dropout: float = 0.1, emb_dropout: float = 0.1,
             freeze: bool = False, **kwargs)

def forward(self, src_embed: Tensor, src_length: Tensor, mask: Tensor = None,
            **kwargs) -> Tuple[Tensor, Tensor]
```
Returns `(x, None)`:
- `x`: float, `(B, S, H)`. Final `LayerNorm` only when `layer_norm == "pre"`.
- `None`: the encoder_hidden slot.

`mask` is `src_mask`, bool `(B, 1, S)`. `src_length` is unused.
`kwargs` gets every field of the batch (see §4) plus `pad=True` in encode mode.

### `RecurrentEncoder` (`encoders.py:31`)
```python
def __init__(self, rnn_type: str = "gru", hidden_size: int = 1, emb_size: int = 1,
             num_layers: int = 1, dropout: float = 0.0, emb_dropout: float = 0.0,
             bidirectional: bool = True, freeze: bool = False, **kwargs)

def forward(self, src_embed: Tensor, src_length: Tensor, mask: Tensor,
            **kwargs) -> Tuple[Tensor, Tensor, Tensor]   # annotation is wrong; returns 2
```
Returns `(output, hidden_concat)`:
- `output`: float `(B, S, dirs*H)`.
- `hidden_concat`: float `(B, dirs*H)`, the last layer's final state.

It needs the batch sorted by `src_length` (descending) because it uses
`pack_padded_sequence`.

### `TransformerDecoder` (`decoders.py:492`)
```python
def __init__(self, num_layers: int = 4, num_heads: int = 8, hidden_size: int = 512,
             ff_size: int = 2048, dropout: float = 0.1, emb_dropout: float = 0.1,
             vocab_size: int = 1, freeze: bool = False, **kwargs)

def forward(self, trg_embed: Tensor, encoder_output: Tensor, encoder_hidden: Tensor,
            src_mask: Tensor, unroll_steps: int, hidden: Tensor, trg_mask: Tensor,
            **kwargs)
```
Returns `(out, x, att, None)`:
- `out`: float `(B, T, V)`, **logits** (no softmax).
- `x`: float `(B, T, H)`, last-layer states.
- `att`: float `(B, T, S)`, cross-attention of the **last layer only**, averaged over
  heads. It is `None` unless `kwargs["return_attention"]` is true.
- `None`: the att_vectors slot.

`encoder_hidden`, `unroll_steps` and `hidden` are ignored. `trg_mask` is required
(asserted). The decoder ANDs it with `subsequent_mask(T)`, which is bool
`(1, T, T)` (`helpers.py:80`).

### `RecurrentDecoder` (`decoders.py:35`)
```python
def __init__(self, rnn_type: str = "gru", emb_size: int = 0, hidden_size: int = 0,
             encoder: Encoder = None, attention: str = "bahdanau", num_layers: int = 1,
             vocab_size: int = 0, dropout: float = 0.0, emb_dropout: float = 0.0,
             hidden_dropout: float = 0.0, init_hidden: str = "bridge",
             input_feeding: bool = True, freeze: bool = False, **kwargs)

def forward(self, trg_embed: Tensor, encoder_output: Tensor, encoder_hidden: Tensor,
            src_mask: Tensor, unroll_steps: int, hidden: Tensor = None,
            prev_att_vector: Tensor = None, **kwargs)
```
Returns `(outputs, hidden, att_probs, att_vectors)`:
- `outputs`: logits `(B, unroll_steps, V)`.
- `hidden`: `(B, layers, H)`, permuted batch-first for DataParallel. For an LSTM it is
  a tuple of two such tensors.
- `att_probs`: `(B, unroll_steps, S)`.
- `att_vectors`: `(B, unroll_steps, H)`.

It loops over time steps internally and can be resumed from `hidden` /
`prev_att_vector`.

---

## 3. How `build_model()` picks the encoder and decoder

`joeynmt/model.py:366-406` (quoted exactly):
```python
    # build encoder
    enc_dropout = enc_cfg.get("dropout", 0.0)
    enc_emb_dropout = enc_cfg["embeddings"].get("dropout", enc_dropout)
    if enc_cfg.get("type", "recurrent") == "transformer":
        assert enc_cfg["embeddings"]["embedding_dim"] == enc_cfg["hidden_size"], (
            "for transformer, emb_size must be "
            "the same as hidden_size"
        )
        emb_size = src_embed.embedding_dim
        encoder = TransformerEncoder(
            **enc_cfg,
            emb_size=emb_size,
            emb_dropout=enc_emb_dropout,
            pad_index=src_pad_index,
        )
    else:
        encoder = RecurrentEncoder(
            **enc_cfg,
            emb_size=src_embed.embedding_dim,
            emb_dropout=enc_emb_dropout,
        )

    # build decoder
    dec_dropout = dec_cfg.get("dropout", 0.0)
    dec_emb_dropout = dec_cfg["embeddings"].get("dropout", dec_dropout)
    if dec_cfg.get("type", "transformer") == "transformer":
        decoder = TransformerDecoder(
            **dec_cfg,
            encoder=encoder,
            vocab_size=len(trg_vocab),
            emb_size=trg_embed.embedding_dim,
            emb_dropout=dec_emb_dropout,
        )
    else:
        decoder = RecurrentDecoder(
            **dec_cfg,
            encoder=encoder,
            vocab_size=len(trg_vocab),
            emb_size=trg_embed.embedding_dim,
            emb_dropout=dec_emb_dropout,
        )
```
Things to watch:
- **It is an if/else, not a registry.** Any `type` other than `"transformer"`
  (so `"conv"` too) currently falls into the **Recurrent** branch without an error.
- **The defaults don't match.** A missing encoder `type` means recurrent. A missing
  decoder `type` means transformer.
- The whole config section is splatted in with `**enc_cfg` / `**dec_cfg`, including
  `type` and the nested `embeddings` dict. Constructors must take `**kwargs`.
- Embeddings are built before this (`model.py:345-364`) as
  `Embeddings(**cfg["embeddings"], vocab_size, padding_idx)`, with optional
  `tied_embeddings`. `Embeddings` has **no positional embeddings**. Its forward is
  `lut(x)`, times `sqrt(E)` if `scale`.
- After `Model(...)` is built:
  1. `tied_softmax` ties `decoder.output_layer.weight` to `trg_embed.lut.weight`.
     The decoder must name its projection **`output_layer`** for this to work.
  2. `initialize_model(model, cfg, ...)` **re-initializes every parameter**
     (`model.py:429`), overwriting any init done in `__init__`.
  3. Pretrained embeddings are loaded, if configured.
- Other places that read the model type:
  - `initialization.py:138`: `cfg["encoder"]["type"] == cfg["decoder"]["type"] == "transformer"`
    (DeepNet alpha/beta, only with `xavier_normal`).
  - `model.py:191,233`: `isinstance(..., TransformerEncoder/Decoder)` to embed prompt masks.
  - `prediction.py:493`: `cfg["model"]["decoder"]["type"] == "transformer"` for the
    attention-plot beam-size check.
  - `search.py:45,54,398`: `isinstance(model.decoder, TransformerDecoder / RecurrentDecoder)`
    (see §5).

---

## 4. Training forward pass: from batch to loss

1. **Data → Batch.** `datasets.py:175 collate_fn` encodes
   - src with `bos=False, eos=True`
   - trg with `bos=True, eos=True`

   then pads to the max length with `pad_index` and builds `Batch`
   (`batch.py:25-81`):
   - `src` long `(B, S)`
   - `src_length` long `(B,)`
   - `src_mask = (src != pad).unsqueeze(1)` → bool `(B, 1, S)`
   - `trg_input = where(trg == eos, pad, trg)[:, :-1]` → long `(B, T)`. Starts with
     BOS; EOS is replaced by pad.
   - `trg = trg[:, 1:]` → long `(B, T)`. Shifted targets, ending in EOS.
   - `trg_mask = (trg != pad).unsqueeze(1)` → bool `(B, 1, T)`
   - `ntokens = trg_mask.sum()`
2. **`TrainManager._train_step`** (`training.py:533`):
   ```python
   self.model.train()
   batch.sort_by_src_length()
   with torch.autocast(**self.autocast):
       batch_loss, _, _, correct_tokens = self.model(return_type="loss", **vars(batch))
   norm_batch_loss = batch.normalize(batch_loss, normalization=..., n_gpu=..., n_accumulation=...)
   norm_batch_loss.backward()          # or scaler.scale(...).backward()
   ```
   `**vars(batch)` passes **every** batch attribute as a kwarg: `src`, `src_length`,
   `src_mask`, `src_prompt_mask`, `trg_input`, `trg`, `trg_mask`, `trg_prompt_mask`,
   `indices`, `nseqs`, `ntokens`, `has_trg`, `is_train`.
3. **`Model.forward(return_type="loss")`** (`model.py:98-122`):
   ```python
   out, _, att_probs, _ = self._encode_decode(**kwargs)
   log_probs = F.log_softmax(out, dim=-1)
   batch_loss = self.loss_function(log_probs, **kwargs)
   n_correct = sum(log_probs.argmax(-1)[trg_mask] == trg[trg_mask])
   return (batch_loss, log_probs, att_probs, n_correct)
   ```
4. **`_encode_decode`** (`model.py:139`):
   - runs `_encode(src, src_length, src_mask, **kwargs)`
   - then `_decode(encoder_output, encoder_hidden, src_mask, trg_input, unroll_steps=T, trg_mask, **kwargs)`
5. **`_encode`** (`model.py:175`):
   `self.encoder(self.src_embed(src), src_length, src_mask, **kwargs)`.
   The encoder gets **embedded** tokens and does its own positional encoding and
   embedding dropout.
6. **`_decode`** (`model.py:201`):
   ```python
   self.decoder(trg_embed=self.trg_embed(trg_input), encoder_output=..., encoder_hidden=...,
                src_mask=src_mask, unroll_steps=unroll_steps, hidden=decoder_hidden,
                prev_att_vector=att_vector, trg_mask=trg_mask, **kwargs)
   ```
   The decoder must return a **4-tuple** with **logits** first.
7. **Loss.** `XentLoss` (`loss.py`):
   - Without smoothing: `NLLLoss(ignore_index=pad, reduction="sum")`.
   - With smoothing: KL divergence against the smoothed targets, pad rows zeroed.

   Then `Batch.normalize` divides by `ntokens` / `nseqs` / 1, then by `n_gpu` and
   `batch_multiplier`.
8. **Update** (`training.py:432-447`):
   - `clip_grad_fun(model.parameters())` (built from `clip_grad_val` / `clip_grad_norm`)
   - `optimizer.step()`
   - `scheduler.step(steps)` if the scheduler steps per step
   - `zero_grad()`

   This runs only every `batch_multiplier` batches.

Validation (`prediction.py:170`) calls the same `return_type="loss"` path under
`no_grad` and also passes `return_prob` and `return_attention`. It then runs
`search()` for hypotheses and BLEU.

---

## 5. `search.py`: how decoding calls the decoder at each step

`search()` (`search.py:825`) first runs `model(return_type="encode", **vars(batch))`
once. Then it dispatches:
- `beam_size < 2` → `greedy(...)`
- otherwise → `beam_search(...)`

### `greedy` dispatch (`search.py:45-61`)
```python
if isinstance(model.decoder, TransformerDecoder): return transformer_greedy(...)
elif isinstance(model.decoder, RecurrentDecoder): return recurrent_greedy(...)
else: raise NotImplementedError(...)
```
Any new decoder class **raises here** until it is added.

### `transformer_greedy` (`search.py:160`)
- `ys` starts as `(B, 1)` BOS.
- `trg_mask = src_mask.new_ones([1, 1, 1])`, a **bool all-ones placeholder**, not
  `(B, 1, T)`.
- Each step:
  ```python
  log_probs, _, att, _ = model(return_type="decode", trg_input=ys, encoder_output=...,
      encoder_hidden=None, src_mask=src_mask, unroll_steps=None, decoder_hidden=None,
      trg_mask=trg_mask, return_attention=return_attn, trg_prompt_mask=...)
  log_probs = log_probs[:, -1]          # keep last position
  ...
  ys = torch.cat([ys, next_word], dim=1)
  ```

### `recurrent_greedy` (`search.py:64`)
Each step feeds only `prev_y` `(B, 1)` with `unroll_steps=1`, and carries
`decoder_hidden=hidden` and `att_vector=prev_att_vector` between calls.

### `beam_search` (`search.py:342`)
Shared setup:
- `is_transformer = isinstance(model.decoder, TransformerDecoder)`
- `encoder_output` and `src_mask` are tiled `k` times.
- For **non**-Transformer decoders it calls `model.decoder._init_hidden(encoder_hidden)`
  and tiles/permutes the result (`search.py:405-418`).

Each step:
- **Transformer branch** (`search.py:508-532`): passes the full `alive_seq`
  `(B*k, t)` as `trg_input` with `unroll_steps=1` and the `[1,1,1]` `trg_mask`, then
  takes `logits[:, -1]`.
- **Else branch** (`search.py:533-555`): passes `alive_seq[:, -1:]` plus
  `decoder_hidden=hidden` and `att_vector=att_vectors`, then
  `logits.squeeze(1)`.

At the end of each step, `encoder_output`, `src_mask`, `hidden` and `att_vectors`
are re-indexed with `select_indices`. For Transformers, beam search returns
attention `None`.

### Answer: full prefix or cache?
**The Transformer decoder re-runs the full prefix every step. There is no
incremental state or KV cache anywhere.**
- `TransformerDecoder.forward` has no cache argument and ignores `hidden`.
- Both search functions pass `decoder_hidden=None` and the whole `ys` / `alive_seq`,
  and slice `[:, -1]`.

Decoding is `O(T²)` decoder passes. It is correct for any causal decoder that
handles a variable-length prefix.

For ConvS2S this means that left-padding by `k-1` and trimming `k-1` works unchanged
at inference. No incremental conv buffers are needed if the conv decoder goes down
the Transformer path.

---

## 6. Padding masks: where they're built and where they go

| Mask | Built | Shape / dtype | Consumers |
|---|---|---|---|
| `src_mask` | `batch.py:55` `(src != pad_index).unsqueeze(1)` | bool `(B, 1, S)`, True = real token (EOS included) | Passed through all the calls below, then used in the attention layers (next rows). |
| ↳ Transformer self-attention | `transformer_layers.py:91` `scores.masked_fill(~mask.unsqueeze(1), -inf)` | broadcast to `(B, heads, S, S)` | Masks **keys** only. Padded query rows are still computed. |
| ↳ Transformer cross-attention | `transformer_layers.py:392` `src_trg_att(memory, memory, h1, mask=src_mask)` | `(B, 1, S)` → `(B, heads, T, S)` | Keys = encoder positions. |
| ↳ RNN attention | `attention.py:82,182` `torch.where(mask > 0, scores, -inf)` | `(B, 1, S)` | Bahdanau / Luong. |
| ↳ RNN encoder | *not used*. Relies on `pack_padded_sequence(src_length)` | | |
| `trg_mask` (train/val) | `batch.py:80` `(trg != pad_index).unsqueeze(1)`, from shifted `trg` | bool `(B, 1, T)`. Lines up position-for-position with `trg_input`, because `trg_input` has EOS→pad. | `Model.forward` (accuracy count), `XentLoss` uses `ignore_index` instead of the mask, and `TransformerDecoder` ANDs it with `subsequent_mask(T)` to get `(B, T, T)` for self-attention. |
| `trg_mask` (search) | `search.py:218, 448` `src_mask.new_ones([1, 1, 1])` | bool `(1, 1, 1)`. Under DataParallel, `(n_dev, 1, 1)`. | ANDed with `subsequent_mask(t)`. **It is a placeholder, not a padding mask.** Search prefixes contain no pad tokens. |
| `subsequent_mask` | `helpers.py:80` `torch.tril(ones(T, T)).unsqueeze(0)` | bool `(1, T, T)` | `TransformerDecoder.forward` only. |
| `src/trg_prompt_mask` | `collate_fn` / `Batch` | long `(B, S)` / `(B, T)` | Embedded and added to the input (Transformer only, gated by `isinstance`). Used for forced decoding in search. Irrelevant for Multi30k. |

Embedding padding: `nn.Embedding(padding_idx=pad)`, and `initialize_model` zeroes
the pad rows again. **Pad positions are still not zero after positional encoding or
any layer.** JoeyNMT never zeroes hidden states at pad positions. Any "mask before
conv" has to happen inside the new module, using
`src_mask.transpose(1, 2)` → `(B, S, 1)`.

---

## 7. Config keys the Transformer encoder reads

From `model.encoder` (`enc_cfg`):

| Key | Read where | Default | Effect |
|---|---|---|---|
| `type` | `model.py:369` | `"recurrent"` | Must be `"transformer"`. Also read with no default at `initialization.py:138` (xavier_normal only). |
| `hidden_size` | `model.py:370` (assert == `embeddings.embedding_dim`), `TransformerEncoder.__init__` | 512 | Model width, also the `_output_size`. Divisible by `num_heads`. |
| `ff_size` | `__init__` | 2048 | FFN inner size. |
| `num_layers` | `__init__`, also `initialization.py:143` | 8 | |
| `num_heads` | `__init__` | 4 | |
| `dropout` | `model.py:367` and `__init__` | 0.0 in `build_model` (the constructor default of 0.1 is never used) | Attention, residual and FFN dropout. Also the fallback for embedding dropout. |
| `freeze` | `__init__` | False | Freezes encoder params. |
| `layer_norm` | `__init__` via `kwargs.get` | `"pre"` for the layers, **but `"post"` for the final LayerNorm check** (`encoders.py:208, 218`) | If the key is omitted, the layers are pre-norm but there is no final norm. |
| `activation` | `kwargs.get` | `"relu"` | FFN activation. |
| `alpha` | `kwargs.get` | 1.0 | Residual weight. Overwritten by DeepNet alpha if `initializer: xavier_normal`. |
| `embeddings.embedding_dim` | `Embeddings`, assert | 64 | |
| `embeddings.scale` | `Embeddings` | False | Multiplies by `sqrt(E)`. |
| `embeddings.freeze` | `Embeddings` | False | |
| `embeddings.dropout` | `model.py:368` → `emb_dropout` | = `encoder.dropout` | Applied after positional encoding. |
| `embeddings.load_pretrained` | `model.py:432` | None | |

`build_model` also injects `emb_size`, `emb_dropout` and `pad_index`. The encoder
takes these through `**kwargs` and ignores `emb_size` and `pad_index`.

Keys at the model level that affect the encoder:
- `tied_embeddings`
- `initializer`, `init_gain`, `init_weight`
- `embed_initializer`, `embed_init_gain`, `embed_init_weight`
- `bias_initializer`, `bias_init_weight`

Under `training`, `scheduling: noam` uses `encoder._output_size`.

---

## Files to touch to add a conv encoder/decoder, and why

Per CLAUDE.md, files under `joeynmt/` may only be edited once they are explicitly
named. This list is the proposal to approve.

**Must change (existing files):**
1. **`joeynmt/encoders.py`**: add `ConvEncoder(Encoder)`.
   - Takes `(src_embed, src_length, mask, **kwargs)`.
   - Returns `(output (B,S,H), None)`. The decoder also needs `z + e`, so either
     return a tuple/stack in the second slot (e.g. `(z, z+e)`) or have the decoder
     recompute it. The second slot is only used by `RecurrentDecoder` and
     `_init_hidden`.
   - Must set `_output_size` (Noam scheduler, decoder sizing).
   - Needs **learned positional embeddings**, because `Embeddings` has none.
   - Scaling encoder gradients by 1/#attn-layers can be a `register_hook` inside the
     module, so `training.py` stays untouched.
2. **`joeynmt/decoders.py`**: add `ConvDecoder(Decoder)` with the Transformer-style
   call signature.
   - Returns `(logits (B,T,V), states, att (B,T,S) or None, None)`.
   - The projection must be named `output_layer` so `tied_softmax` works.
   - **Must not assume `trg_mask` is `(B,1,T)`**: search passes `(1,1,1)`.
   - Causality comes from left-padding plus trimming, so running over the full prefix
     is correct.
3. **`joeynmt/model.py`**: in `build_model`, add `elif ... == "conv"` branches
   **before** the `else` fallbacks, and import the new classes. Without this,
   `type: conv` silently builds an RNN. The prompt-mask `isinstance` checks in
   `_encode` / `_decode` can stay as they are.
4. **`joeynmt/search.py`**: add the new decoder at two places.
   - `greedy` (line 45): send `ConvDecoder` to `transformer_greedy`. Otherwise it
     raises `NotImplementedError`.
   - `beam_search` (line 398): treat `ConvDecoder` as full-prefix,
     e.g. `isinstance(model.decoder, (TransformerDecoder, ConvDecoder))`. Otherwise it
     calls `_init_hidden` and feeds only the last token, which is wrong for a conv
     decoder.
5. **`joeynmt/initialization.py`**: needed for the paper's init. `initialize_model`
   runs after construction and overwrites everything:
   - params whose name contains `"embed"` get the embed initializer
   - `"bias"` params get the bias initializer
   - everything else with 2+ dims (conv weights too) gets `initializer`

   Two options:
   - (a) add a `conv`-aware branch here for N(0, sqrt(4p/n)) on GLU convs and
     N(0, sqrt(1/n)) elsewhere
   - (b) call a `ConvEncoder/ConvDecoder.reset_parameters()` from `build_model`
     after `initialize_model`

   (b) touches only `model.py`. Either way, avoid `"embed"` in names of non-embedding
   params. Line 138 also indexes `cfg["encoder"]["type"]` with no default.

**Optional (existing files):**
6. `joeynmt/prediction.py:493`: the attention-plot guard checks
   `decoder.type == "transformer"`. Add `"conv"` if attention plots are wanted with
   beam search disabled. Not required for training or BLEU.

**New files (no baseline code affected):**
7. `joeynmt/conv_layers.py` (optional, new): GLU conv block, residual scaling by
   sqrt(0.5), and the multi-step attention layer. This mirrors
   `transformer_layers.py` and keeps `encoders.py` / `decoders.py` small.
8. `configs/conv_multi30k_deen.yaml`, plus Multi30k configs for the RNN and
   Transformer baselines (new files; existing configs stay untouched).
9. `test/unit/test_conv_encoder.py`, `test_conv_decoder.py`, and a search/model-build
   test. These should cover:
   - output shapes
   - length preservation
   - causality (changing a future token leaves earlier outputs unchanged)
   - pad positions zeroed before conv
   - `type: conv` builds the conv classes
   - greedy and beam search run
10. `EXPERIMENTS.md` (does not exist yet) and `REPORT_NOTES.md`: required logging
    under CLAUDE.md.

**Don't need to change:**
- `training.py`: generic over `Model`. Clipping, Nesterov and lr come from config.
- `batch.py`, `datasets.py`, `loss.py`: masks, shifting and loss don't depend on the
  architecture.
- `embeddings.py`: positional embeddings live in the new modules.
- `hub_interface.py`: goes through `search()`.
- `transformer_layers.py`, `attention.py`: baselines, left untouched.
