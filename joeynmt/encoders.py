# coding: utf-8
"""
Various encoders
"""
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from joeynmt.conv_layers import (
    RESIDUAL_SCALE,
    GLUConvBlock,
    init_default_layer_,
    init_embedding_,
)
from joeynmt.helpers import freeze_params
from joeynmt.transformer_layers import PositionalEncoding, TransformerEncoderLayer


class Encoder(nn.Module):
    """
    Base encoder class
    """

    # pylint: disable=abstract-method
    @property
    def output_size(self):
        """
        Return the output size

        :return:
        """
        return self._output_size


class RecurrentEncoder(Encoder):
    """Encodes a sequence of word embeddings"""

    # pylint: disable=unused-argument
    def __init__(
        self,
        rnn_type: str = "gru",
        hidden_size: int = 1,
        emb_size: int = 1,
        num_layers: int = 1,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        bidirectional: bool = True,
        freeze: bool = False,
        **kwargs,
    ) -> None:
        """
        Create a new recurrent encoder.

        :param rnn_type: RNN type: `gru` or `lstm`.
        :param hidden_size: Size of each RNN.
        :param emb_size: Size of the word embeddings.
        :param num_layers: Number of encoder RNN layers.
        :param dropout:  Is applied between RNN layers.
        :param emb_dropout: Is applied to the RNN input (word embeddings).
        :param bidirectional: Use a bi-directional RNN.
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super().__init__()

        self.emb_dropout = torch.nn.Dropout(p=emb_dropout, inplace=False)
        self.type = rnn_type
        self.emb_size = emb_size

        rnn = nn.GRU if rnn_type == "gru" else nn.LSTM

        self.rnn = rnn(
            emb_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self._output_size = 2 * hidden_size if bidirectional else hidden_size

        if freeze:
            freeze_params(self)

    def _check_shapes_input_forward(
        self, src_embed: Tensor, src_length: Tensor, mask: Tensor
    ) -> None:
        """
        Make sure the shape of the inputs to `self.forward` are correct.
        Same input semantics as `self.forward`.

        :param src_embed: embedded source tokens
        :param src_length: source length
        :param mask: source mask
        """
        # pylint: disable=unused-argument
        assert src_embed.shape[0] == src_length.shape[0]
        assert src_embed.shape[2] == self.emb_size
        # assert mask.shape == src_embed.shape
        assert len(src_length.shape) == 1

    def forward(self, src_embed: Tensor, src_length: Tensor, mask: Tensor,
                **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Applies a bidirectional RNN to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param src_embed: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :param kwargs:
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """
        self._check_shapes_input_forward(
            src_embed=src_embed, src_length=src_length, mask=mask
        )
        total_length = src_embed.size(1)

        # apply dropout to the rnn input
        src_embed = self.emb_dropout(src_embed)

        packed = pack_padded_sequence(src_embed, src_length.cpu(), batch_first=True)
        output, hidden = self.rnn(packed)

        if isinstance(hidden, tuple):
            hidden, memory_cell = hidden  # pylint: disable=unused-variable

        output, _ = pad_packed_sequence(
            output, batch_first=True, total_length=total_length
        )
        # hidden: dir*layers x batch x hidden
        # output: batch x max_length x directions*hidden
        batch_size = hidden.size()[1]
        # separate final hidden states by layer and direction
        hidden_layerwise = hidden.view(
            self.rnn.num_layers,
            2 if self.rnn.bidirectional else 1,
            batch_size,
            self.rnn.hidden_size,
        )
        # final_layers: layers x directions x batch x hidden

        # concatenate the final states of the last layer for each directions
        # thanks to pack_padded_sequence final states don't include padding
        fwd_hidden_last = hidden_layerwise[-1:, 0]
        bwd_hidden_last = hidden_layerwise[-1:, 1]

        # only feed the final state of the top-most layer to the decoder
        # pylint: disable=no-member
        hidden_concat = torch.cat([fwd_hidden_last, bwd_hidden_last], dim=2).squeeze(0)
        # final: batch x directions*hidden

        assert hidden_concat.size(0) == output.size(0), (
            hidden_concat.size(),
            output.size(),
        )
        return output, hidden_concat

    def __repr__(self):
        return f"{self.__class__.__name__}(rnn={self.rnn})"


class TransformerEncoder(Encoder):
    """
    Transformer Encoder
    """

    def __init__(
        self,
        hidden_size: int = 512,
        ff_size: int = 2048,
        num_layers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        freeze: bool = False,
        **kwargs,
    ):
        """
        Initializes the Transformer.
        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super().__init__()

        self._output_size = hidden_size

        # build all (num_layers) layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                size=hidden_size,
                ff_size=ff_size,
                num_heads=num_heads,
                dropout=dropout,
                alpha=kwargs.get("alpha", 1.0),
                layer_norm=kwargs.get("layer_norm", "pre"),
                activation=kwargs.get("activation", "relu"),
            ) for _ in range(num_layers)
        ])

        self.pe = PositionalEncoding(hidden_size)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self.layer_norm = (
            nn.LayerNorm(hidden_size, eps=1e-6)
            if kwargs.get("layer_norm", "post") == "pre" else None
        )

        if freeze:
            freeze_params(self)

    def forward(
        self,
        src_embed: Tensor,
        src_length: Tensor,  # unused
        mask: Tensor = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param src_embed: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, 1, src_len)
        :param kwargs:
        :return:
            - output: hidden states with shape (batch_size, max_length, hidden)
            - None
        """
        # pylint: disable=unused-argument
        x = self.pe(src_embed)  # add position encoding to word embeddings
        if kwargs.get("src_prompt_mask", None) is not None:  # add src_prompt_mask
            x = x + kwargs["src_prompt_mask"]
        x = self.emb_dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x, None

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(num_layers={len(self.layers)}, "
            f"num_heads={self.layers[0].src_src_att.num_heads}, "
            f"alpha={self.layers[0].alpha}, "
            f'layer_norm="{self.layers[0]._layer_norm_position}", '
            f"activation={self.layers[0].feed_forward.pwff_layer[1]})"
        )


class ConvEncoder(Encoder):
    """
    Convolutional encoder (Gehring et al., 2017). New class: the RNN and
    Transformer encoders above are untouched.

    Returns ``(x, None)`` where ``x`` is the concatenation of the two tensors the
    multi-step attention of ``ConvDecoder`` needs::

        x = cat([z_u, z_c], dim=-1)          (batch, src_len, 2 * emb_size)
        z_u = conv stack output              attention keys
        z_c = sqrt(0.5) * (z_u + e)          attention values, the "+ e_j" term

    They travel in one tensor because ``search.py`` hardcodes
    ``encoder_hidden=None`` on the Transformer decode path (search.py:242, 518,
    543), and ``encoder_output`` is the only tensor beam search tiles and
    re-indexes. See SPEC.md §2.1.

    Note ``output_size`` is ``hidden_size`` (the conv width, what the Noam
    scheduler wants at training.py:121), not the width of the returned tensor;
    use ``attention_size`` for the latter's halves.
    """

    # pylint: disable=unused-argument,too-many-instance-attributes
    def __init__(
        self,
        hidden_size: int = 512,
        emb_size: int = 512,
        num_layers: int = 8,
        kernel_width: int = 3,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        freeze: bool = False,
        **kwargs,
    ) -> None:
        """
        Create a convolutional encoder.

        :param hidden_size: conv channel width H
        :param emb_size: embedding / attention-space width E
        :param num_layers: number of GLU conv blocks
        :param kernel_width: conv kernel width k (odd)
        :param dropout: dropout on every conv input
        :param emb_dropout: dropout after the positional embeddings
        :param freeze: freeze the parameters of this encoder
        :param kwargs: `max_position` (default 1024) and everything else
            `build_model` splats in
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.kernel_width = kernel_width
        self.dropout_p = dropout
        self.max_position = kwargs.get("max_position", 1024)

        # Embeddings has no positional embeddings, so they live here (SPEC.md §6 D)
        self.pos_embed = nn.Embedding(self.max_position, emb_size)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        # projections only when the widths differ
        self.input_proj = (
            nn.Linear(emb_size, hidden_size) if emb_size != hidden_size else None
        )
        self.layers = nn.ModuleList([
            GLUConvBlock(
                hidden_size=hidden_size,
                kernel_width=kernel_width,
                dropout=dropout,
                padding_mode="symmetric",
            ) for _ in range(num_layers)
        ])
        self.output_proj = (
            nn.Linear(hidden_size, emb_size) if emb_size != hidden_size else None
        )

        # set by build_model once the decoder is known; None disables scaling
        self.num_attention_layers = None

        self._output_size = hidden_size
        self.attention_size = emb_size

        self.reset_parameters()
        if freeze:
            freeze_params(self)

    def reset_parameters(self) -> None:
        """
        ConvS2S initialisation (SPEC.md §5). Called again from `build_model`
        after `initialize_model`, which would otherwise overwrite everything.
        """
        init_embedding_(self.pos_embed)
        if self.input_proj is not None:
            init_default_layer_(self.input_proj, fan_in=self.emb_size)
        for layer in self.layers:
            layer.reset_parameters()
        if self.output_proj is not None:
            init_default_layer_(self.output_proj, fan_in=self.hidden_size)

    def forward(
        self,
        src_embed: Tensor,
        src_length: Tensor,
        mask: Tensor = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply the conv stack to the embedded source.

        :param src_embed: embedded source, shape (batch, src_len, emb_size)
        :param src_length: source lengths, shape (batch,). Unused.
        :param mask: source mask, shape (batch, 1, src_len), True = real token
        :return:
            - x: (batch, src_len, 2 * emb_size), see the class docstring
            - None: the encoder_hidden slot
        """
        _, src_len, _ = src_embed.size()
        assert src_len <= self.max_position, \
            f"src_len {src_len} exceeds max_position {self.max_position}"

        # (E2)-(E3) learned positional embeddings, then embedding dropout
        pos = torch.arange(src_len, device=src_embed.device).unsqueeze(0)
        emb = self.emb_dropout(src_embed + self.pos_embed(pos))

        # (E4) float pad mask (batch, src_len, 1), 1.0 = real token
        pad_mask = None
        if mask is not None:
            pad_mask = mask.transpose(1, 2).to(emb.dtype)
            emb = emb * pad_mask

        # (E5)-(E9) conv stack; each block zeroes pads before its convolution
        x = emb if self.input_proj is None else self.input_proj(emb)
        for layer in self.layers:
            x = layer(x, pad_mask)

        # (E10)-(E13) attention keys and values
        z_u = x if self.output_proj is None else self.output_proj(x)
        z_c = RESIDUAL_SCALE * (z_u + emb)
        if pad_mask is not None:
            z_u = z_u * pad_mask
            z_c = z_c * pad_mask

        # (E14)
        x = torch.cat([z_u, z_c], dim=-1)

        # (E15) scale encoder gradients by 1 / num_attention_layers
        if self.num_attention_layers is not None and x.requires_grad:
            scale = 1.0 / self.num_attention_layers
            x.register_hook(lambda grad: grad * scale)

        return x, None

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(num_layers={len(self.layers)}, "
            f"hidden_size={self.hidden_size}, emb_size={self.emb_size}, "
            f"kernel_width={self.kernel_width}, "
            f"num_attention_layers={self.num_attention_layers})"
        )
