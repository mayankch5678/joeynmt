#!/usr/bin/env python
# coding: utf-8
"""
Report parameter counts for one or more configs via `build_model`, without
touching the training data or running a step.

The real vocabulary file is used when it exists; otherwise a synthetic
vocabulary of `--vocab-size` entries stands in. As long as the configs share an
embedding dim and tie nothing, the vocabulary-dependent parameters
(2 embedding tables + the output layer) are identical across them, so the
comparison does not depend on that number -- the "shared" column shows it.

Usage:
    python scripts/count_params.py configs/multi30k_*.yaml
    python scripts/count_params.py configs/multi30k_conv.yaml --vocab-size 8000
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from joeynmt.config import load_config
from joeynmt.model import build_model
from joeynmt.vocabulary import Vocabulary

SPECIALS = {
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "sep_token": None,
    "unk_id": 0,
    "pad_id": 1,
    "bos_id": 2,
    "eos_id": 3,
    "sep_id": None,
    "lang_tags": [],
}


def make_vocab(voc_file: Path, fallback_size: int):
    """Real vocabulary if the file exists, else a synthetic one."""
    if voc_file is not None and voc_file.is_file():
        tokens = [t for t in voc_file.read_text(encoding="utf-8").split("\n") if t]
        source = str(voc_file)
    else:
        tokens = [f"t{i}" for i in range(fallback_size - 4)]
        source = f"synthetic ({fallback_size} types)"
    return Vocabulary(tokens=tokens, cfg=SimpleNamespace(**SPECIALS)), source


def count(config_path: Path, fallback_size: int, seed: int = 42):
    cfg = load_config(config_path)
    voc_file = cfg["data"]["src"].get("voc_file")
    vocab, source = make_vocab(Path(voc_file) if voc_file else None, fallback_size)

    torch.manual_seed(seed)
    model = build_model(cfg["model"], src_vocab=vocab, trg_vocab=vocab)

    # exact names: `MultiHeadedAttention` also owns a submodule called
    # `output_layer`, so a substring match would fold attention projections
    # into the vocabulary-dependent total
    shared_names = {
        "src_embed.lut.weight",
        "trg_embed.lut.weight",
        "decoder.output_layer.weight",
    }
    named = list(model.named_parameters())
    total = sum(p.numel() for _, p in named)
    shared = sum(p.numel() for n, p in named if n in shared_names)
    encoder = sum(p.numel() for n, p in named if n.startswith("encoder."))
    decoder = sum(
        p.numel() for n, p in named
        if n.startswith("decoder.") and n not in shared_names
    )
    return {
        "name": config_path.stem,
        "vocab": len(vocab),
        "source": source,
        "total": total,
        "shared": shared,
        "body": total - shared,
        "encoder": encoder,
        "decoder": decoder,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("configs", nargs="+", type=Path)
    ap.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="synthetic vocabulary size when voc_file is absent"
    )
    args = ap.parse_args()

    rows = [count(c, args.vocab_size) for c in args.configs]

    print(f"\nvocabulary: {rows[0]['vocab']} types  [{rows[0]['source']}]")
    header = f"{'config':<26}{'total':>12}{'shared':>12}{'body':>12}" \
             f"{'encoder':>12}{'decoder':>12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:<26}{r['total']:>12,}{r['shared']:>12,}{r['body']:>12,}"
            f"{r['encoder']:>12,}{r['decoder']:>12,}"
        )

    if len(rows) > 1:
        for key in ("total", "body"):
            values = [r[key] for r in rows]
            spread = (max(values) - min(values)) / min(values)
            print(
                f"\nspread over {key:<6}: "
                f"{min(values):,} .. {max(values):,}  ({100 * spread:.1f}%)"
            )
            base = min(values)
            for r in rows:
                print(f"  {r['name']:<26}{100 * (r[key] - base) / base:+7.1f}%")


if __name__ == "__main__":
    main()
