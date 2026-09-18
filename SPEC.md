# SPEC.md — ConvS2S (Gehring et al. 2017) for JoeyNMT v2.3

Equations, tensor shapes and interface signatures only.
Sources: Gehring et al. 2017 §3; `CODEBASE_MAP.md` §2, §4, §5, §6.

---

## 0. Symbols

| Symbol | Meaning |
|---|---|
| `B` | batch size |
| `S` | padded source length |
| `T` | padded target length (`trg_input` length) |
| `E` | embedding dim = attention-space dim = `embeddings.embedding_dim` |
| `H` | conv hidden size (channels) = `hidden_size` |
| `V` | target vocab size |
| `k` | kernel width (odd) |
| `L_e`, `L_d` | number of encoder / decoder layers |
| `P` | max positions in the learned positional table |
| `p` | retain probability `= 1 - dropout` |
| `n` | fan-in of a weight (Linear: `in_features`; Conv1d: `in_channels * k`) |
| `σ` | logistic sigmoid |
| `⊙` | elementwise product |
| `m` | `src_mask.transpose(1, 2)` ∈ {0,1}^(B,S,1) |
| `m_j` | per-sentence source token count `src_mask.sum(-1)` ∈ ℕ^(B,1,1) |

---

## 1. GLU block

Input `u ∈ ℝ^(B,C,N)`, weight `W ∈ ℝ^(2C,C,k)`, bias `b ∈ ℝ^(2C)`:

```
v      = Conv1d_{C→2C,k}(u; W, b)          v ∈ ℝ^(B,2C,N')
(A, G) = split(v, C, dim=1)                A, G ∈ ℝ^(B,C,N')
GLU(v) = A ⊙ σ(G)                          ∈ ℝ^(B,C,N')
```

---

## 2. Encoder

### 2.1 Equations

```
(E1)  w_j   = Lut_src(x_j)                                   (B,S,E)   [given as src_embed]
(E2)  e     = w + Pos_enc[0:S]                Pos_enc ∈ ℝ^(P,E)        (B,S,E)
(E3)  e     ← Dropout_emb(e)                                 (B,S,E)
(E4)  z^0   = e W_in^e + b_in^e               W_in^e ∈ ℝ^(E,H)         (B,S,H)

      for l = 1 … L_e:
(E5)    r     = z^{l-1} ⊙ m                                  (B,S,H)
(E6)    u     = Dropout(r)ᵀ                                  (B,H,S)
(E7)    u_pad = pad(u, left=(k-1)/2, right=(k-1)/2)          (B,H,S+k-1)
(E8)    y     = GLU(Conv1d_{H→2H,k}(u_pad))ᵀ                 (B,S,H)
(E9)    z^l   = √0.5 · (y + z^{l-1})                         (B,S,H)

(E10) z^u   = z^{L_e} W_out^e + b_out^e       W_out^e ∈ ℝ^(H,E)        (B,S,E)
(E11) z^u   ← z^u ⊙ m                                        (B,S,E)
(E12) z^c   = √0.5 · (z^u + e)                               (B,S,E)
(E13) z^c   ← z^c ⊙ m                                        (B,S,E)
(E14) encoder_output = cat([z^u, z^c], dim=-1)               (B,S,2E)
(E15) grad hook on z^u, z^c:   g ↦ g / L_d
```

`(E7)` preserves length: `S + (k-1) - k + 1 = S`.
`(E4)` and `(E10)` are the identity (no parameters) when `E == H`; the
projections exist only when `emb_size != hidden_size`. Same for `(D3)`/`(D17)`.
`(E15)` implements "encoder gradients scaled by 1/num_attention_layers";
the number of attention layers equals `L_d` (multi-step attention, §3.2).

`z^u` = attention **keys**, `z^c` = attention **values**. They are concatenated
into a single tensor because `search.py` hardcodes `encoder_hidden=None` on the
Transformer decode path (`search.py:242, 518, 543`), so the `encoder_hidden`
slot cannot carry `z^c`. `encoder_output` is the only tensor that search tiles
and re-indexes for beam search.

### 2.2 Shape table

| Step | Tensor | Shape |
|---|---|---|
| in | `src_embed` | `(B, S, E)` |
| in | `src_length` | `(B,)` |
| in | `mask` (`src_mask`) | `(B, 1, S)` bool |
| E2 | `Pos_enc.weight` | `(P, E)` |
| E4 | `W_in^e`, `b_in^e` | `(E, H)`, `(H,)` |
| E4 | `z^0` | `(B, S, H)` |
| E6 | `u` | `(B, H, S)` |
| E7 | `u_pad` | `(B, H, S+k-1)` |
| E8 | conv weight, bias | `(2H, H, k)`, `(2H,)` |
| E8 | pre-GLU | `(B, 2H, S)` |
| E8 | `y` | `(B, S, H)` |
| E9 | `z^l` | `(B, S, H)` |
| E10 | `W_out^e`, `b_out^e` | `(H, E)`, `(E,)` |
| E10 | `z^u` | `(B, S, E)` |
| E12 | `z^c` | `(B, S, E)` |
| E14 | `encoder_output` | `(B, S, 2E)` |
| out | return | `(encoder_output, None)` |

`self._output_size = H`  (read by `training.py:121` for the Noam scheduler).

---

## 3. Decoder

### 3.1 Equations

```
(D1)  g     = trg_embed + Pos_dec[0:T]        Pos_dec ∈ ℝ^(P,E)        (B,T,E)
(D2)  g     ← Dropout_emb(g)                                 (B,T,E)
(D3)  h^0   = g W_in^d + b_in^d               W_in^d ∈ ℝ^(E,H)         (B,T,H)
(D4)  z^u, z^c = split(encoder_output, E, dim=-1)            (B,S,E) each

      for l = 1 … L_d:
(D5)    u     = Dropout(h^{l-1} ⊙ m_t)ᵀ                      (B,H,T)
(D6)    u_pad = pad(u, left=k-1, right=0)                    (B,H,T+k-1)
(D7)    h̃     = GLU(Conv1d_{H→2H,k}(u_pad))ᵀ                 (B,T,H)
(D8)    d^l   = h̃ W_d^l + b_d^l + g          W_d^l ∈ ℝ^(H,E)          (B,T,E)
(D9)    s^l   = d^l (z^u)ᵀ                                   (B,T,S)
(D10)   s^l   ← masked_fill(s^l, ¬src_mask, -inf)            (B,T,S)
(D11)   a^l   = softmax(s^l, dim=-1)                         (B,T,S)
(D12)   c^l   = a^l z^c                                      (B,T,E)
(D13)   c^l   ← c^l · m_j · √(1/m_j)                         (B,T,E)
(D14)   c^l_h = c^l W_c^l + b_c^l            W_c^l ∈ ℝ^(E,H)          (B,T,H)
(D15)   o^l   = √0.5 · (h̃ + c^l_h)                           (B,T,H)
(D16)   h^l   = √0.5 · (o^l + h^{l-1})                       (B,T,H)

(D17) t      = h^{L_d} W_out^d + b_out^d      W_out^d ∈ ℝ^(H,E)        (B,T,E)
(D18) t      ← Dropout(t)                                    (B,T,E)
(D19) logits = output_layer(t)               W_o ∈ ℝ^(V,E), no bias    (B,T,V)
```

`(D6)`+conv gives `T + (k-1) - k + 1 = T`, i.e. output position `i` sees inputs
`i-k+1 … i` only. The equivalent formulation in `CLAUDE.md` — `Conv1d(padding=k-1)`
on both sides producing `(B, 2H, T+k-1)`, then trimming the last `k-1` columns —
yields the identical tensor.

Causality comes from `(D6)` alone. `trg_mask` is **not** usable for it:
`search.py:218, 448` pass `src_mask.new_ones([1,1,1])`.

`m_t` in `(D5)`: `trg_mask.transpose(1,2)` ∈ {0,1}^(B,T,1) when
`trg_mask.size(-1) == T` (train/validation); the identity otherwise (search,
where `trg_mask` is `(1,1,1)` and prefixes contain no pad).

`(D13)`: `m_j` = number of non-pad source positions per sentence,
broadcast `(B,1,1)`; restores the scale lost by averaging over `m_j` inputs.

Attention is computed in **every** layer `l = 1 … L_d` (multi-step attention).

### 3.2 Shape table

| Step | Tensor | Shape |
|---|---|---|
| in | `trg_embed` | `(B, T, E)` |
| in | `encoder_output` | `(B, S, 2E)` |
| in | `encoder_hidden` | `None` (ignored) |
| in | `src_mask` | `(B, 1, S)` bool |
| in | `unroll_steps` | `int` or `None` (ignored) |
| in | `hidden` | `None` (ignored) |
| in | `trg_mask` | `(B, 1, T)` train, `(1, 1, 1)` search |
| D1 | `Pos_dec.weight` | `(P, E)` |
| D3 | `W_in^d`, `b_in^d` | `(E, H)`, `(H,)` |
| D3 | `h^0` | `(B, T, H)` |
| D4 | `z^u`, `z^c` | `(B, S, E)` each |
| D6 | `u_pad` | `(B, H, T+k-1)` |
| D7 | conv weight, bias | `(2H, H, k)`, `(2H,)` |
| D7 | pre-GLU | `(B, 2H, T)` |
| D7 | `h̃` | `(B, T, H)` |
| D8 | `W_d^l`, `b_d^l` | `(H, E)`, `(E,)` |
| D8 | `d^l` | `(B, T, E)` |
| D11 | `a^l` | `(B, T, S)` |
| D12 | `c^l` | `(B, T, E)` |
| D14 | `W_c^l`, `b_c^l` | `(E, H)`, `(H,)` |
| D16 | `h^l` | `(B, T, H)` |
| D17 | `W_out^d`, `b_out^d` | `(H, E)`, `(E,)` |
| D19 | `output_layer.weight` | `(V, E)`, `bias=False` (tied_softmax) |
| D19 | `logits` | `(B, T, V)` |
| out | return | `(logits, h^{L_d}, att, None)` |

`att = a^{L_d}` ∈ `(B, T, S)` when `kwargs["return_attention"]` is true, else `None`.
`self._output_size = V`  (read by `search.py:394`).
The projection must be named `output_layer` for `tied_softmax`; this requires `E`
to equal `embeddings.embedding_dim` of the target embedding.

---

## 4. Where each √0.5 applies

| # | Location | Sum being halved in variance |
|---|---|---|
| 1 | `(E9)` | encoder conv-block residual: GLU output + block input |
| 2 | `(E12)` | encoder attention values: `z^u + e` |
| 3 | `(D15)` | decoder attention residual: `h̃ + c^l_h` |
| 4 | `(D16)` | decoder conv-block residual: attention output + block input |

No √0.5 on `(D8)`; see §6 note A.

Because of #3, a decoder conv block cannot close its own residual: attention is
applied to `h̃` between the GLU `(D7)` and the block residual `(D16)`.
`GLUConvBlock(residual=False)` therefore returns the bare GLU output and
`ConvDecoder` performs `(D16)` itself. The encoder uses `residual=True`, where
`(E9)` is the only residual in the block.

---

## 5. Initialisation

Applied by `reset_parameters()` called **after** `initialize_model()`
(`model.py:429`), which would otherwise overwrite everything.

| Parameter group | Distribution | `n` |
|---|---|---|
| `Lut_src`, `Lut_trg`, `Pos_enc`, `Pos_dec` | `N(0, 0.1)` | — |
| Conv weights feeding a GLU: `(E8)`, `(D7)` | `N(0, √(4p/n))` | `H · k` |
| `W_in^e` `(E4)`, `W_out^e` `(E10)` | `N(0, √(1/n))` | `E`, `H` |
| `W_in^d` `(D3)`, `W_out^d` `(D17)` | `N(0, √(1/n))` | `E`, `H` |
| `W_d^l` `(D8)`, `W_c^l` `(D14)` | `N(0, √(1/n))` | `H`, `E` |
| `output_layer.weight` `(D19)` | `N(0, √(1/n))` | `E` |
| `Pos_*` positional tables | `N(0, 0.1)` | — |
| all biases | `0` | — |
| `Lut_src[pad]`, `Lut_trg[pad]` | `0` | — |

`p = 1 - dropout` of the dropout applied to that layer's **input**
(`(E6)`, `(D5)` for the convs).

Parameter names for the conv/projection weights must not contain the substring
`"embed"` (`initialization.py` dispatches on it).

---

## 6. Paper ambiguities, resolved

- **A.** `(D8)` follows the paper literally: `d = W h̃ + b + g`, no scaling.
  fairseq's reference implementation uses `√0.5 · (W h̃ + b + g)`. Chosen: paper form,
  per `CLAUDE.md`.
- **B.** Non-GLU layers use `N(0, √(1/n))` per `CLAUDE.md`. The paper's dropout-aware
  form is `N(0, √(p/n))`.
- **C.** `(E12)` applies `√0.5` to `z^u + e`, consistent with rule "residual sums
  scaled by √0.5"; the paper writes the context as `Σ_j a_ij (z_j + e_j)` unscaled.
- **D.** Positional embeddings are learned tables of size `(P, E)` (paper §3.1);
  `Embeddings` in JoeyNMT has none, so they live inside `ConvEncoder`/`ConvDecoder`.
- **E.** Returned `att` is the last layer's `a^{L_d}`, matching
  `TransformerDecoder`'s "last layer only" convention.

---

## 7. Forward signatures

```python
class ConvEncoder(Encoder):

    def __init__(
        self,
        hidden_size: int = 512,      # H
        emb_size: int = 512,         # E
        num_layers: int = 8,         # L_e
        kernel_width: int = 3,       # k, odd
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        freeze: bool = False,
        **kwargs,                    # max_position: int = 1024
    ) -> None: ...

    def reset_parameters(self) -> None: ...   # §5, re-run after initialize_model

    def forward(
        self,
        src_embed: Tensor,          # (B, S, E)
        src_length: Tensor,         # (B,)      unused
        mask: Tensor = None,        # (B, 1, S) bool
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:     # ((B, S, 2E), None)
        ...

    # attributes
    _output_size: int               # = H  (Noam, training.py:121)
    attention_size: int             # = E  (width of each half of the output)
    num_attention_layers: int|None  # = L_d, set by build_model; None = no scaling


class ConvDecoder(Decoder):

    def __init__(
        self,
        hidden_size: int = 512,      # H
        emb_size: int = 512,         # E, must equal encoder.attention_size
        num_layers: int = 8,         # L_d
        kernel_width: int = 3,       # k
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        vocab_size: int = 1,         # V
        freeze: bool = False,
        **kwargs,                    # max_position: int = 1024, encoder: Encoder
    ) -> None: ...

    def reset_parameters(self) -> None: ...   # §5, re-run after initialize_model

    def forward(
        self,
        trg_embed: Tensor,            # (B, T, E)
        encoder_output: Tensor,       # (B, S, 2E)
        encoder_hidden: Tensor = None,  # unused
        src_mask: Tensor = None,      # (B, 1, S) bool, required
        unroll_steps: int = None,     # unused
        hidden: Tensor = None,        # unused
        trg_mask: Tensor = None,      # (B, 1, T) train | (1, 1, 1) search
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # ((B, T, V), (B, T, H), (B, T, S), None)
        ...

    # attributes
    _output_size: int               # = V  (search.py:394)
    output_layer: nn.Linear         # (E -> V), bias=False, named for tied_softmax
    attentions: nn.ModuleList       # L_d MultiStepAttention, one per layer
    layer_attentions: List[Tensor]  # L_d detached (B, T, S) maps, report figures
```

Both constructors receive the whole config section splatted (`**enc_cfg` /
`**dec_cfg`, including `type` and the nested `embeddings` dict) plus injected
`emb_size`, `emb_dropout`, `pad_index`, `vocab_size`, `encoder`
(`model.py:366-406`), hence the mandatory `**kwargs`.

Constraint: `E == embeddings.embedding_dim`, `k` odd, `P ≥ max(S, T)`.
