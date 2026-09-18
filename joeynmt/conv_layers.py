# coding: utf-8
"""
Convolutional building blocks for ConvS2S (Gehring et al., 2017).

New file: nothing here is used by the RNN or Transformer baselines.
Equation labels (E*) / (D*) refer to SPEC.md.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import weight_norm

# sqrt(0.5): halves the variance of a sum of two residual streams (SPEC.md §4)
RESIDUAL_SCALE = math.sqrt(0.5)


def glu_init_std(fan_in: int, dropout: float = 0.0) -> float:
    """
    N(0, sqrt(4p/n)) for layers whose output is fed to a GLU (SPEC.md §5).

    :param fan_in: number of input connections per output unit
    :param dropout: dropout rate applied to this layer's *input*; p = 1 - dropout
    """
    return math.sqrt(4.0 * (1.0 - dropout) / fan_in)


def default_init_std(fan_in: int) -> float:
    """N(0, sqrt(1/n)) for every other layer (SPEC.md §5)."""
    return math.sqrt(1.0 / fan_in)


def _set_weight(module: nn.Module, new_weight: Tensor) -> None:
    """
    Write `new_weight` into `module.weight`, transparently handling a
    weight-normalised module (assignment goes through the parametrization's
    right_inverse, which recomputes g = ||v|| so that weight == new_weight).
    """
    with torch.no_grad():
        if parametrize.is_parametrized(module, "weight"):
            module.weight = new_weight
        else:
            module.weight.copy_(new_weight)


def init_glu_layer_(module: nn.Module, fan_in: int, dropout: float = 0.0) -> None:
    """In-place N(0, sqrt(4p/n)) init of a Conv1d/Linear that feeds a GLU."""
    std = glu_init_std(fan_in, dropout)
    _set_weight(module, torch.randn_like(module.weight) * std)
    if getattr(module, "bias", None) is not None:
        nn.init.zeros_(module.bias)


def init_default_layer_(module: nn.Module, fan_in: int) -> None:
    """In-place N(0, sqrt(1/n)) init of a Conv1d/Linear that does not feed a GLU."""
    std = default_init_std(fan_in)
    _set_weight(module, torch.randn_like(module.weight) * std)
    if getattr(module, "bias", None) is not None:
        nn.init.zeros_(module.bias)


def init_embedding_(emb: nn.Embedding, std: float = 0.1) -> None:
    """In-place N(0, 0.1) init of an embedding table; pad row zeroed."""
    nn.init.normal_(emb.weight, mean=0.0, std=std)
    if emb.padding_idx is not None:
        with torch.no_grad():
            emb.weight[emb.padding_idx].zero_()


class GLUConvBlock(nn.Module):
    """
    One ConvS2S convolution block.

        y = GLU(Conv1d_{H->2H,k}(pad(dropout(x * mask))))
        out = sqrt(0.5) * (y + x)

    Padding modes:
      - "symmetric": (k-1)/2 on both sides, length preserved (encoder, E7).
      - "causal": (k-1) on both sides, then the last (k-1) columns are trimmed,
        so position i sees inputs i-k+1 .. i only (decoder, D6). Identical to
        left-padding by (k-1) with no right padding.

    With ``residual=False`` the block returns the bare GLU output, unscaled. The
    decoder needs this because its attention is applied between the GLU and the
    residual add (D15)-(D16), so the block cannot close the residual itself.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_width: int,
        dropout: float = 0.0,
        padding_mode: str = "symmetric",
        use_weight_norm: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()
        assert padding_mode in ("symmetric", "causal"), padding_mode
        assert kernel_width >= 1, kernel_width
        if padding_mode == "symmetric":
            assert kernel_width % 2 == 1, \
                f"symmetric padding needs an odd kernel_width, got {kernel_width}"

        self.hidden_size = hidden_size
        self.kernel_width = kernel_width
        self.padding_mode = padding_mode
        self.dropout_p = dropout
        self.use_weight_norm = use_weight_norm
        self.residual = residual

        self.dropout = nn.Dropout(p=dropout)
        conv = nn.Conv1d(hidden_size, 2 * hidden_size, kernel_width, bias=True)
        self.conv = weight_norm(conv, name="weight", dim=0) if use_weight_norm else conv

        if padding_mode == "symmetric":
            half = (kernel_width - 1) // 2
            self.pad = (half, half)
            self.trim = 0
        else:
            self.pad = (kernel_width - 1, kernel_width - 1)
            self.trim = kernel_width - 1

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Conv output feeds a GLU -> N(0, sqrt(4p/n)), n = H * k."""
        init_glu_layer_(
            self.conv,
            fan_in=self.hidden_size * self.kernel_width,
            dropout=self.dropout_p,
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        :param x: (batch, length, hidden_size)
        :param mask: (batch, length, 1) float, 1.0 = real token, 0.0 = pad.
            Padded positions are zeroed before the convolution.
        :return: (batch, length, hidden_size); the bare GLU output if
            `residual=False`
        """
        residual = x
        if mask is not None:
            x = x * mask
        x = self.dropout(x)

        x = x.transpose(1, 2)  # (B, H, N)
        x = F.pad(x, self.pad)  # pylint: disable=not-callable
        x = self.conv(x)  # (B, 2H, N + trim)
        if self.trim > 0:
            x = x[:, :, :-self.trim]  # causality: drop the future tail
        x = F.glu(x, dim=1)  # (B, H, N)
        x = x.transpose(1, 2)  # (B, N, H)

        if not self.residual:
            return x
        return (x + residual) * RESIDUAL_SCALE

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(hidden_size={self.hidden_size}, "
            f"kernel_width={self.kernel_width}, "
            f'padding_mode="{self.padding_mode}", '
            f"dropout={self.dropout_p}, weight_norm={self.use_weight_norm}, "
            f"residual={self.residual})"
        )


class MultiStepAttention(nn.Module):
    """
    One attention step of ConvS2S multi-step attention (SPEC.md D8-D15).

    Every decoder layer owns one of these, so the decoder attends to the source
    `num_layers` times, once per layer::

        d = W_d h + b_d + g                     query, g = target embedding
        a = softmax(d z_u^T)                    over non-pad source positions
        c = (a z_c) * sqrt(m)                   m = real source tokens/sentence
        out = sqrt(0.5) * (h + W_c c + b_c)

    `z_u` are the attention keys and `z_c = sqrt(0.5)(z_u + e)` the values, i.e.
    the "+ e_j" of the paper; `ConvEncoder` returns both.
    """

    def __init__(self, hidden_size: int, emb_size: int) -> None:
        """
        :param hidden_size: conv channel width H
        :param emb_size: embedding / attention-space width E
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.query_proj = nn.Linear(hidden_size, emb_size)
        self.out_proj = nn.Linear(emb_size, hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Neither projection feeds a GLU -> N(0, sqrt(1/n)) (SPEC.md §5)."""
        init_default_layer_(self.query_proj, fan_in=self.hidden_size)
        init_default_layer_(self.out_proj, fan_in=self.emb_size)

    def forward(
        self,
        x: Tensor,
        trg_emb: Tensor,
        keys: Tensor,
        values: Tensor,
        src_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        :param x: post-GLU decoder states, (batch, trg_len, hidden_size)
        :param trg_emb: target embeddings g, (batch, trg_len, emb_size)
        :param keys: z_u, (batch, src_len, emb_size)
        :param values: z_c, (batch, src_len, emb_size)
        :param src_mask: (batch, 1, src_len) bool, True = real token
        :return:
            - (batch, trg_len, hidden_size)
            - attention probabilities, (batch, trg_len, src_len)
        """
        residual = x

        # (D8) query: no sqrt(0.5) here, paper form (SPEC.md §6 A)
        query = self.query_proj(x) + trg_emb

        # (D9)-(D11)
        scores = torch.bmm(query, keys.transpose(1, 2))
        scores = scores.masked_fill(~src_mask, float("-inf"))
        attention = torch.softmax(scores, dim=-1)

        # (D12)-(D13): m * sqrt(1/m) == sqrt(m), undoing the averaging over m inputs
        context = torch.bmm(attention, values)
        n_src = src_mask.sum(dim=-1, keepdim=True).to(context.dtype)
        context = context * n_src.sqrt()

        # (D14)-(D15)
        context = self.out_proj(context)
        return RESIDUAL_SCALE * (residual + context), attention

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(hidden_size={self.hidden_size}, "
            f"emb_size={self.emb_size})"
        )
