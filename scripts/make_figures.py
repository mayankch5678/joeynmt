#!/usr/bin/env python
# coding: utf-8
"""
Generate the report figures into IMG/ (ACL two-column style, vector PDF).

    fig_validation_bleu.pdf   greedy dev BLEU vs step, all models      (1 column)
    fig_attention_layers.pdf  per-layer cross-attention of the conv model (2 columns)
    fig_bleu_by_length.pdf    BLEU bucketed by source length           (1 column)
    fig_architecture.pdf      ConvS2S schematic                        (1 column)

Run from the repository root (the conv model is rebuilt from its config, whose
data paths are relative to it):

    python scripts/make_figures.py

Nothing is invented or interpolated: every plotted number is parsed from the
logs / hypothesis files below and printed to stdout so it can be checked against
EXPERIMENTS.md. A figure whose inputs are missing is skipped with a list of
exactly what is missing; the other figures are still produced.

Inputs (all under --models-dir, one sub-directory per model):
    <model>/train.log            fig 1   ("Evaluation result (greedy): bleu: X")
    <model>/validations.txt      fig 1   optional, exact step numbers
    <model>/best.hyps.test       fig 3   detokenized beam-5 test hypotheses
    <conv-model>/best.ckpt       fig 2
    <conv-model>/config.yaml     fig 2   (falls back to --conv-config)
plus, from --data-dir: test.de, test.en, bpe.10000.codes, vocab.txt (fig 3).
"""
import argparse
import itertools
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

COL1 = 3.15  # ACL single column, inches
COL2 = 6.5  # ACL double column, inches

# (model dir, legend label, colour, linestyle, marker). Colours are muted and differ
# in lightness; linestyle and marker differ too, so they survive greyscale.
MODELS = [
    ("multi30k_conv", "ConvS2S", "#1f3a5f", "-", "o"),
    ("multi30k_conv_lastattn", "ConvS2S (last-layer attn)", "#7f9fc4", "--", "s"),
    ("multi30k_transformer", "Transformer", "#a4531b", "-.", "^"),
    ("multi30k_rnn", "RNN", "#555555", ":", "D"),
]

# BLEU-by-length buckets in source BPE tokens: (low, high inclusive, label)
BUCKETS = [
    (1, 10, "1–10"),
    (11, 15, "11–15"),
    (16, 20, "16–20"),
    (21, 25, "21–25"),
    (26, None, "26+"),
]

SERIF_CANDIDATES = [
    "Times New Roman",
    "Times",
    "Liberation Serif",  # metric-compatible with Times New Roman, common on Linux
    "Nimbus Roman",
    "Nimbus Roman No9 L",
    "TeX Gyre Termes",
    "FreeSerif",
    "STIXGeneral",  # ships with matplotlib, Times-like
    "DejaVu Serif",  # ships with matplotlib, always there
]


class MissingInputs(Exception):
    """Raised by a figure when files it needs are not there."""

    def __init__(self, missing: List[str]):
        super().__init__("; ".join(missing))
        self.missing = missing


# ----------------------------------------------------------------------------
# style
# ----------------------------------------------------------------------------


def pick_serif_font() -> str:
    """First installed font from SERIF_CANDIDATES (no warnings for missing ones)."""
    for name in SERIF_CANDIDATES:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except ValueError:
            continue
    return "DejaVu Serif"


def set_style() -> str:
    font = pick_serif_font()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [font, "DejaVu Serif"],
        "mathtext.fontset": "stix",  # Times-like maths whatever the text font is
        "font.size": 8,
        "axes.labelsize": 9,  # axis titles
        "axes.titlesize": 9,
        "axes.labelweight": "normal",
        "axes.titleweight": "normal",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        "legend.frameon": False,
        "legend.handlelength": 2.4,
        "pdf.fonttype": 42,  # embed TrueType, no Type 3 (venues reject those)
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
    return font


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> wrote {path}")


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ----------------------------------------------------------------------------
# figure 1: greedy dev BLEU vs step
# ----------------------------------------------------------------------------

GREEDY_RE = re.compile(r"Evaluation result \(greedy\):.*?\bbleu:\s*(-?\d+(?:\.\d+)?)")
STEP_RE = re.compile(r"Epoch\s+\d+,\s+Step:\s+(\d+),")
VALIDATIONS_STEP_RE = re.compile(r"Steps:\s*(\d+)")
VALIDATIONS_BLEU_RE = re.compile(r"\bbleu:\s*(-?\d+(?:\.\d+)?)")


def read_validation_curve(model_dir: Path) -> Tuple[List[Tuple[int, float]], str]:
    """
    BLEU values come from train.log ("Evaluation result (greedy)" lines). Those
    lines carry no step number, so the step is taken from validations.txt when it
    exists and agrees with the log value by value; otherwise from the last
    "Step:" progress line seen before the evaluation.

    :return: ([(step, bleu)], description of where the steps came from)
    :raises MissingInputs:
    """
    log = model_dir / "train.log"
    if not log.is_file():
        raise MissingInputs([f"{log}  (training log)"])

    evals: List[Tuple[Optional[int], float]] = []
    last_step = None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = STEP_RE.search(line)
        if m:
            last_step = int(m.group(1))
            continue
        m = GREEDY_RE.search(line)
        if m:
            evals.append((last_step, float(m.group(1))))
    if not evals:
        raise MissingInputs([
            f"{log} has no 'Evaluation result (greedy): ... bleu: X' lines"
        ])

    vfile = model_dir / "validations.txt"
    if vfile.is_file():
        pairs = []
        for line in vfile.read_text(encoding="utf-8").splitlines():
            ms, mb = VALIDATIONS_STEP_RE.search(line), VALIDATIONS_BLEU_RE.search(line)
            if ms and mb:
                pairs.append((int(ms.group(1)), float(mb.group(1))))
        agree = len(pairs) == len(evals) and all(
            abs(b_v - b_l) < 0.01 for (_, b_v), (_, b_l) in zip(pairs, evals)
        )
        if agree:
            return pairs, "steps from validations.txt (BLEU matches train.log)"
        print(
            f"  WARNING {vfile}: {len(pairs)} entries vs {len(evals)} greedy lines "
            "in train.log, or BLEU values differ; ignoring validations.txt and "
            "using the 'Step:' lines of train.log"
        )

    if any(step is None for step, _ in evals):
        raise MissingInputs([
            f"{log}: an evaluation precedes any 'Step:' line, so its step is "
            f"unknown and {vfile} is absent"
        ])
    return [(int(s), b) for s, b in evals
            ], "steps from the last 'Step:' line before each evaluation"


def fig_validation_bleu(args) -> None:
    banner("fig_validation_bleu.pdf: greedy dev BLEU vs training step")
    curves = []
    missing = []
    for dirname, label, colour, ls, marker in MODELS:
        try:
            points, source = read_validation_curve(args.models_dir / dirname)
        except MissingInputs as e:
            if dirname == "multi30k_conv_lastattn":  # optional per the brief
                print(f"  {label}: not plotted, missing {e.missing}")
            else:
                missing += e.missing
            continue
        curves.append((dirname, label, colour, ls, marker, points))
        steps, bleus = zip(*points)
        best_step, best = max(points, key=lambda p: p[1])
        print(f"  {label} [{dirname}]: {len(points)} evaluations, {source}")
        print(f"    best greedy dev BLEU {best:.2f} at step {best_step}; "
              f"last {bleus[-1]:.2f} at step {steps[-1]}")
        print("    step:bleu  " + " ".join(f"{s}:{b:.2f}" for s, b in points))
    if missing:
        raise MissingInputs(missing)
    if not curves:
        raise MissingInputs(["no model directory with a train.log under "
                             f"{args.models_dir}"])

    fig, ax = plt.subplots(figsize=(COL1, 2.35))
    for _, label, colour, ls, marker, points in curves:
        steps, bleus = zip(*points)
        ax.plot(
            steps,
            bleus,
            color=colour,
            linestyle=ls,
            marker=marker,
            markevery=max(1, len(points) // 8),
            label=label,
        )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Dev BLEU (greedy)")
    ax.set_xlim(left=0)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}k" if x else "0")
    )
    ax.grid(axis="y")
    ax.legend(loc="lower right")
    save(fig, args.out_dir, "fig_validation_bleu.pdf")


# ----------------------------------------------------------------------------
# figure 2: per-layer cross-attention
# ----------------------------------------------------------------------------


def _load_yaml(path: Path) -> Dict:
    import yaml
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fig_attention_layers(args) -> None:
    banner("fig_attention_layers.pdf: per-layer attention of the conv decoder")
    model_dir = args.models_dir / args.conv_model
    ckpt = model_dir / "best.ckpt"
    config = model_dir / "config.yaml"
    if not config.is_file():
        config = Path(args.conv_config)

    missing = []
    if not ckpt.is_file():
        missing.append(f"{ckpt}  (best checkpoint)")
    if not config.is_file():
        missing.append(f"{model_dir / 'config.yaml'} and {args.conv_config}  (config)")
    if missing:
        raise MissingInputs(missing)

    cfg = _load_yaml(config)
    data = cfg.get("data", {})
    for side in ("src", "trg"):
        side_cfg = data.get(side, {})
        voc = side_cfg.get("voc_file")
        if voc and not Path(voc).is_file():
            missing.append(f"{voc}  ({side} vocabulary; run from repo root)")
        codes = side_cfg.get("tokenizer_cfg", {}).get("codes")
        if codes and not Path(codes).is_file():
            missing.append(f"{codes}  ({side} BPE codes; run from repo root)")
        if data.get("dev") and side_cfg.get("lang"):
            dev_file = Path(f"{data['dev']}.{side_cfg['lang']}")
            if not dev_file.is_file():
                missing.append(
                    f"{dev_file}  (dev {side_cfg['lang']}; run from repo root)"
                )
    if missing:
        raise MissingInputs(sorted(set(missing)))

    try:
        import torch

        from joeynmt.config import parse_global_args
        from joeynmt.decoders import ConvDecoder
        from joeynmt.prediction import prepare
        from joeynmt.search import search
    except ImportError as e:
        raise MissingInputs([f"python package needed for this figure: {e}"])

    cfg["model_dir"] = str(model_dir)  # ckpt is resolved from here
    print(f"  config {config}, checkpoint {ckpt}")
    jargs = parse_global_args(cfg, rank=0, mode="test")
    model, _, dev_data, _ = prepare(jargs, rank=0, mode="test")
    if not isinstance(model.decoder, ConvDecoder):
        raise MissingInputs([f"{config} does not build a ConvDecoder "
                             f"(got {type(model.decoder).__name__})"])
    model.eval()

    # one dev sentence, as a batch of one, in dataset order
    dev_data.reset_indices(random_subset=-1)
    loader = dev_data.make_iter(
        batch_size=1,
        batch_type="sentence",
        shuffle=False,
        seed=dev_data.seed,
        num_workers=0,
        eos_index=model.eos_index,
        pad_index=model.pad_index,
        device=jargs.device,
    )
    batch = next(itertools.islice(loader, args.sentence_index, None), None)
    if batch is None or int(batch.indices[0]) != args.sentence_index:
        raise MissingInputs([
            f"dev sentence {args.sentence_index} (dev set has {len(dev_data)})"
        ])

    # greedy decode, then teacher-force the hypothesis: the decoder is causal, so
    # the attention of row t is the one used when generating token t.
    out, _, _ = search(
        model=model,
        batch=batch,
        max_output_length=jargs.test.max_output_length,
        beam_size=1,
        beam_alpha=jargs.test.beam_alpha,
        n_best=1,
        autocast=jargs.autocast,
        return_attention=False,
        return_prob="none",
        generate_unk=False,
        repetition_penalty=-1,
        no_repeat_ngram_size=-1,
    )
    ids = out[0].tolist()
    if model.eos_index in ids:
        ids = ids[:ids.index(model.eos_index) + 1]
    trg_tokens = model.trg_vocab.arrays_to_sentences(
        np.array([ids]), cut_at_eos=True
    )[0]
    src_tokens = model.src_vocab.arrays_to_sentences(
        batch.src.cpu().numpy(), cut_at_eos=False
    )[0]
    trg_in = torch.tensor([[model.bos_index] + ids[:-1]], device=jargs.device)

    with torch.no_grad(), torch.autocast(**jargs.autocast):
        enc_out, enc_hid, _, _ = model(return_type="encode", **vars(batch))
        model(
            return_type="decode",
            encoder_output=enc_out,
            encoder_hidden=enc_hid,
            src_mask=batch.src_mask,
            trg_input=trg_in,
            unroll_steps=trg_in.size(1),
            trg_mask=None,
        )
    maps = [a[0].float().cpu().numpy() for a in model.decoder.layer_attentions]
    reference = dev_data.get_list(lang=dev_data.trg_lang, tokenized=False)[
        args.sentence_index]

    print(f"  dev sentence {args.sentence_index}, greedy decoding, "
          f"{len(maps)} decoder layers, attention maps (T={len(trg_tokens)}, "
          f"S={len(src_tokens)})")
    print(f"  source tokens   : {' '.join(src_tokens)}")
    print(f"  hypothesis (BPE): {' '.join(trg_tokens)}")
    print(f"  reference (raw) : {reference}")
    for layer, att in enumerate(maps, start=1):
        assert att.shape == (len(trg_tokens), len(src_tokens)), att.shape
        sums = att.sum(-1)
        if not np.allclose(sums, 1.0, atol=1e-2):
            print(f"  WARNING layer {layer}: attention rows do not sum to 1 "
                  f"(min {sums.min():.4f}, max {sums.max():.4f})")
        print(f"  layer {layer}: rows = target tokens, columns = source tokens")
        with np.printoptions(precision=2, suppress=True, linewidth=200):
            print(att)
        top = att.argmax(-1)
        print("    argmax per target token: " + ", ".join(
            f"{t}->{src_tokens[j]} ({att[i, j]:.2f})"
            for i, (t, j) in enumerate(zip(trg_tokens, top))
        ))

    def _esc(tokens):  # matplotlib treats $ as mathtext
        return [t.replace("$", r"\$") for t in tokens]

    n = len(maps)
    fig, axes = plt.subplots(
        1, n, figsize=(COL2, 2.6), sharey=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for layer, (ax, att) in enumerate(zip(axes, maps), start=1):
        mesh = ax.pcolormesh(
            att,
            cmap="Greys",
            vmin=0.0,
            vmax=1.0,
            edgecolors="face",
            linewidth=0.3,
        )
        ax.set_xticks(np.arange(len(src_tokens)) + 0.5)
        ax.set_xticklabels(_esc(src_tokens), rotation=90)
        ax.set_yticks(np.arange(len(trg_tokens)) + 0.5)
        ax.set_yticklabels(_esc(trg_tokens))
        ax.set_xlim(0, len(src_tokens))
        ax.set_ylim(len(trg_tokens), 0)  # first target token on top
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.set_xlabel(f"Layer {layer}")
    axes[0].set_ylabel("Generated token")
    cbar = fig.colorbar(mesh, ax=list(axes), fraction=0.02, pad=0.01)
    cbar.ax.tick_params(labelsize=8, width=0.5, length=2)
    cbar.outline.set_linewidth(0.5)
    save(fig, args.out_dir, "fig_attention_layers.pdf")


# ----------------------------------------------------------------------------
# figure 3: BLEU by source length
# ----------------------------------------------------------------------------


def _bucket_index(length: int) -> int:
    for i, (lo, hi, _) in enumerate(BUCKETS):
        if length >= lo and (hi is None or length <= hi):
            return i
    return 0  # length 0 (empty source) falls into the first bucket


def fig_bleu_by_length(args) -> None:
    banner("fig_bleu_by_length.pdf: BLEU bucketed by source length")
    split = args.split
    src_file = args.data_dir / f"{split}.{args.src_lang}"
    ref_file = args.data_dir / f"{split}.{args.trg_lang}"
    codes_file = args.bpe_codes or args.data_dir / "bpe.10000.codes"
    vocab_file = args.vocab or args.data_dir / "vocab.txt"

    missing = [
        f"{p}  ({what})" for p, what in (
            (src_file, "source sentences"),
            (ref_file, "references"),
            (codes_file, "BPE codes"),
            (vocab_file, "BPE vocabulary"),
        ) if not p.is_file()
    ]
    hyp_files = {}
    for dirname, label, *_ in MODELS:
        hyp = args.models_dir / dirname / f"best.hyps.{split}"
        if hyp.is_file():
            hyp_files[dirname] = hyp
        elif dirname == "multi30k_conv_lastattn":
            print(f"  {label}: not plotted, missing {hyp}")
        else:
            model_config = f"configs/{dirname}.yaml"
            missing.append(
                f"{hyp}  ({label} hypotheses; written by training, or by "
                f"`python -m joeynmt test {model_config} "
                f"--output-path {args.models_dir / dirname / 'best.hyps'}`)"
            )
    if missing:
        raise MissingInputs(missing)

    try:
        from sacrebleu.metrics import BLEU
        from subword_nmt.apply_bpe import BPE
    except ImportError as e:
        raise MissingInputs([f"python package needed for this figure: {e}"])

    def _lines(path: Path) -> List[str]:
        lines = path.read_text(encoding="utf-8").split("\n")
        if lines and lines[-1] == "":  # trailing newline
            lines.pop()
        return lines

    src_lines, refs = _lines(src_file), _lines(ref_file)
    if len(src_lines) != len(refs):
        raise MissingInputs([f"{src_file} has {len(src_lines)} lines, {ref_file} "
                             f"has {len(refs)}"])

    # source length in BPE tokens, segmented as during training (same codes, and
    # restricted to the vocabulary as SubwordNMTTokenizer.set_vocab does)
    with codes_file.open("r", encoding="utf-8") as f:
        bpe = BPE(f, -1, "@@", None, None)
    bpe.vocab = set(vocab_file.read_text(encoding="utf-8").split("\n")) - {""}
    lengths = [len(bpe.process_line(" ".join(s.split())).split()) for s in src_lines]
    bucket_of = [_bucket_index(n) for n in lengths]
    counts = [bucket_of.count(i) for i in range(len(BUCKETS))]
    print(f"  {len(refs)} {split} sentences; source BPE length min {min(lengths)}, "
          f"mean {np.mean(lengths):.1f}, max {max(lengths)}")
    print("  bucket sizes: " + ", ".join(
        f"{b[2]}: n={c}" for b, c in zip(BUCKETS, counts)))

    corpus_bleu = BLEU(tokenize="13a")
    sent_bleu = BLEU(tokenize="13a", effective_order=True)
    results = {}
    for dirname, label, *_ in MODELS:
        if dirname not in hyp_files:
            continue
        hyps = _lines(hyp_files[dirname])
        if len(hyps) != len(refs):
            raise MissingInputs([
                f"{hyp_files[dirname]} has {len(hyps)} lines, expected {len(refs)}"
            ])
        overall = corpus_bleu.corpus_score(hyps, [refs]).score
        sent_scores = np.array([
            sent_bleu.sentence_score(h, [r]).score for h, r in zip(hyps, refs)
        ])
        per_sent, per_corpus = [], []
        for i in range(len(BUCKETS)):
            idx = [j for j, b in enumerate(bucket_of) if b == i]
            if not idx:
                per_sent.append(float("nan"))
                per_corpus.append(float("nan"))
                continue
            per_sent.append(float(sent_scores[idx].mean()))
            per_corpus.append(
                corpus_bleu.corpus_score([hyps[j] for j in idx],
                                         [[refs[j] for j in idx]]).score)
        results[dirname] = (per_sent, per_corpus)
        print(f"  {label} [{hyp_files[dirname]}]")
        print(f"    corpus BLEU over the whole {split} set: {overall:.2f} "
              "(compare with EXPERIMENTS.md)")
        print("    bucket             " + " ".join(f"{b[2]:>7}" for b in BUCKETS))
        print("    mean sentence BLEU " + " ".join(f"{v:7.2f}" for v in per_sent))
        print("    corpus BLEU        " + " ".join(f"{v:7.2f}" for v in per_corpus))

    which = 0 if args.length_metric == "sentence" else 1
    ylabel = ("Sentence-level BLEU (mean)"
              if args.length_metric == "sentence" else "Corpus BLEU")
    fig, ax = plt.subplots(figsize=(COL1, 2.35))
    x = np.arange(len(BUCKETS))
    for dirname, label, colour, ls, marker in MODELS:
        if dirname in results:
            ax.plot(x, results[dirname][which], color=colour, linestyle=ls,
                    marker=marker, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b[2]}\n({c})" for b, c in zip(BUCKETS, counts)])
    ax.set_xlabel("Source length (BPE tokens); n sentences in brackets")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y")
    ax.legend(loc="lower left")
    save(fig, args.out_dir, "fig_bleu_by_length.pdf")


# ----------------------------------------------------------------------------
# figure 4: architecture schematic
# ----------------------------------------------------------------------------


def fig_architecture(args) -> None:
    banner("fig_architecture.pdf: ConvS2S schematic (no data)")
    xmin, xmax, ymin, ymax = -0.25, 10.55, -1.0, 14.8
    fig = plt.figure(figsize=(COL1, COL1 * (ymax - ymin) / (xmax - xmin)))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis("off")

    lw = 0.6

    def box(x0, y0, w, h, text):
        ax.add_patch(Rectangle((x0, y0), w, h, fill=False, linewidth=lw,
                               edgecolor="black"))
        ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=8,
                linespacing=1.1)

    def arrow(p0, p1):
        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(
            arrowstyle="-|>", lw=lw, color="black", shrinkA=0, shrinkB=0,
            mutation_scale=6))

    def line(*pts):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color="black", linewidth=lw, solid_capstyle="butt")

    ex0, dx0, w = 0.9, 5.75, 4.0  # encoder / decoder box left edge, box width
    ecx, dcx = ex0 + w / 2, dx0 + w / 2
    bus_x, g_x = 5.25, 10.0

    # column headings and inputs
    ax.text(ecx, -0.6, "Encoder", ha="center", va="center", fontsize=9)
    ax.text(dcx, -0.6, "Decoder", ha="center", va="center", fontsize=9)
    ax.text(ecx, 0.5, "source tokens", ha="center", va="center")
    ax.text(dcx, 0.5, "target prefix", ha="center", va="center")

    # embeddings
    for x0, cx in ((ex0, ecx), (dx0, dcx)):
        arrow((cx, 0.9), (cx, 1.6))
        box(x0, 1.6, w, 1.3, "Word + position\nembeddings")

    # encoder: GLU conv stack, then keys z^u and values z^c
    arrow((ecx, 2.9), (ecx, 3.5))
    box(ex0, 3.5, w, 1.5, "GLU conv + residual\n$\\times L_e$")
    arrow((ecx, 5.0), (ecx, 6.4))
    box(ex0, 6.4, w, 1.9, "encoder output\nkeys $z^u$\nvalues $z^c$")
    enc_out_y = 7.35
    # e: input embeddings added to the values
    line((ex0, 2.25), (0.4, 2.25), (0.4, enc_out_y))
    arrow((0.4, enc_out_y), (ex0, enc_out_y))
    ax.text(0.15, 4.8, "$e$", ha="center", va="center")

    # decoder: L_d layers, each a causal GLU conv then attention
    arrow((dcx, 2.9), (dcx, 3.5))
    attn_y = []
    y0 = 3.5
    for layer in range(3):
        box(dx0, y0, w, 1.0, "GLU conv (causal)")
        arrow((dcx, y0 + 1.0), (dcx, y0 + 1.6))
        box(dx0, y0 + 1.6, w, 1.0, "Attention")
        attn_y.append(y0 + 2.1)
        if layer < 2:
            arrow((dcx, y0 + 2.6), (dcx, y0 + 3.05))
        y0 += 3.05
    top = y0 - 3.05 + 2.6
    arrow((dcx, top), (dcx, top + 0.6))
    box(dx0, top + 0.6, w, 1.0, "Linear + softmax")
    arrow((dcx, top + 1.6), (dcx, top + 2.1))
    ax.text(dcx, top + 2.4, "output distribution", ha="center", va="center")

    # multi-step attention: the encoder output is read by EVERY decoder layer
    line((ex0 + w, enc_out_y), (bus_x, enc_out_y))
    line((bus_x, attn_y[0]), (bus_x, attn_y[-1]))
    for y in attn_y:
        arrow((bus_x, y), (dx0, y))

    # g: the target embeddings enter every attention query
    line((dx0 + w, 2.25), (g_x, 2.25), (g_x, attn_y[-1]))
    for y in attn_y:
        arrow((g_x, y), (dx0 + w, y))
    ax.text(g_x + 0.25, 7.0, "$g$", ha="center", va="center")

    print("  schematic only: 3 decoder layers drawn (configs use L_d = 3, L_e = 4)")
    save(fig, args.out_dir, "fig_architecture.pdf")


# ----------------------------------------------------------------------------

FIGURES = {
    "1": ("fig_validation_bleu", fig_validation_bleu),
    "2": ("fig_attention_layers", fig_attention_layers),
    "3": ("fig_bleu_by_length", fig_bleu_by_length),
    "4": ("fig_architecture", fig_architecture),
}


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate the report figures (vector PDF, ACL style)."
    )
    ap.add_argument("--models-dir", type=Path, default=Path("models"),
                    help="directory holding one sub-directory per trained model")
    ap.add_argument("--out-dir", type=Path, default=Path("IMG"),
                    help="where the PDFs are written")
    ap.add_argument("--figures", default="1,2,3,4",
                    help="comma-separated subset of 1,2,3,4 (default: all)")
    ap.add_argument("--data-dir", type=Path, default=Path("test/data/multi30k"),
                    help="Multi30k dir (fig 3: test.de/.en, BPE codes, vocab)")
    ap.add_argument(
        "--split", default="test", help="split used by fig 3 (best.hyps.<split>)"
    )
    ap.add_argument("--src-lang", default="de")
    ap.add_argument("--trg-lang", default="en")
    ap.add_argument("--bpe-codes", type=Path, default=None,
                    help="default: <data-dir>/bpe.10000.codes")
    ap.add_argument("--vocab", type=Path, default=None,
                    help="default: <data-dir>/vocab.txt")
    ap.add_argument("--length-metric", choices=["sentence", "corpus"],
                    default="sentence",
                    help="fig 3 y-axis: mean sentence-level BLEU (default) or "
                    "corpus BLEU per bucket; both are printed either way")
    ap.add_argument("--conv-model", default="multi30k_conv",
                    help="model directory used for the attention figure")
    ap.add_argument("--conv-config", default="configs/multi30k_conv.yaml",
                    help="used if <conv-model>/config.yaml is missing")
    ap.add_argument("--sentence-index", type=int, default=0,
                    help="dev sentence decoded for the attention figure")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    sys.path.insert(0, str(REPO_ROOT))
    font = set_style()
    print(f"font: {font} (fallbacks: DejaVu Serif)")

    wanted = [f.strip() for f in args.figures.split(",") if f.strip()]
    unknown = [f for f in wanted if f not in FIGURES]
    if unknown:
        print(f"unknown figure id(s) {unknown}; choose from {sorted(FIGURES)}")
        return 2

    made, skipped, failed = [], [], []
    for fid in wanted:
        name, func = FIGURES[fid]
        try:
            func(args)
            made.append(name)
        except MissingInputs as e:
            print(f"  SKIPPED {name}: missing input(s):")
            for item in e.missing:
                print(f"    - {item}")
            skipped.append(name)
        except Exception as e:  # keep going: one broken figure must not stop the rest
            print(f"  FAILED {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed.append(name)

    banner("summary")
    print(f"  written : {', '.join(made) or '-'}")
    print(f"  skipped : {', '.join(skipped) or '-'}")
    print(f"  failed  : {', '.join(failed) or '-'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
