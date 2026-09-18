#!/usr/bin/env python
# coding: utf-8
"""
Overfit sanity check for the ConvS2S model.

Takes the first N sentence pairs of `test/data/toy/train` and uses them as both
the training and the validation set, then trains until the step cap. A correct
seq2seq model must be able to memorise a handful of pairs: if this does not
converge, the architecture is broken, not the hyperparameters.

Success criteria (asserted):
  * training loss below `--max-loss` per token
  * greedy BLEU on the same pairs above `--min-bleu`

Validation in JoeyNMT always decodes greedily, and dev == train here, so the
BLEU reported per validation is exactly the greedy BLEU on the trained pairs.

Usage:
    python scripts/overfit_test.py
    python scripts/overfit_test.py --config configs/conv_overfit.yaml -n 50
"""
import argparse
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
SCORE_RE = re.compile(r"(\w+):\s*(-?[\d.]+(?:[eE][-+]?\d+)?)")


def build_subset(
    source_prefix: Path, out_prefix: Path, n: int, langs: List[str]
) -> int:
    """Write the first `n` lines of each side to `out_prefix`.<lang>."""
    counts = set()
    for lang in langs:
        src = source_prefix.with_suffix(f".{lang}")
        assert src.is_file(), f"missing {src}"
        lines = src.read_text(encoding="utf-8").splitlines()[:n]
        counts.add(len(lines))
        out = out_prefix.with_suffix(f".{lang}")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {len(lines):4d} lines -> {out.relative_to(ROOT)}")
    assert len(counts) == 1, f"sides differ in length: {counts}"
    return counts.pop()


def parse_validations(path: Path) -> List[Dict[str, float]]:
    """Parse `validations.txt` into one dict per validation point."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        scores = {k: float(v) for k, v in SCORE_RE.findall(line)}
        if "Steps" not in scores:
            continue
        # per-token loss: prediction.py computes ppl = exp(total_loss/ntokens)
        if "ppl" in scores and scores["ppl"] > 0:
            scores["loss_per_token"] = math.log(scores["ppl"])
        rows.append(scores)
    return rows


def describe_plateau(rows: List[Dict[str, float]], key: str) -> str:
    """Where did `key` stop improving?"""
    values = [(int(r["Steps"]), r[key]) for r in rows if key in r]
    if not values:
        return f"no {key} recorded"
    best_step, best = min(values, key=lambda kv: kv[1]) if key == "loss_per_token" \
        else max(values, key=lambda kv: kv[1])
    last_step, last = values[-1]
    return (
        f"best {key} {best:.4f} at step {best_step}; "
        f"last {last:.4f} at step {last_step}"
    )


def main() -> int:
    # pylint: disable=too-many-locals
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/conv_overfit.yaml")
    parser.add_argument("-n", "--num-pairs", type=int, default=50)
    parser.add_argument("--source", default="test/data/toy/train")
    parser.add_argument("--subset", default="test/data/toy/overfit50")
    parser.add_argument(
        "--max-loss",
        type=float,
        default=0.1,
        help="training loss per token must fall below this"
    )
    parser.add_argument(
        "--min-bleu",
        type=float,
        default=95.0,
        help="greedy BLEU on the same pairs must exceed this"
    )
    parser.add_argument("--keep-model-dir", action="store_true")
    args = parser.parse_args()

    config = ROOT / args.config
    assert config.is_file(), f"missing {config}"

    print(f"== building the {args.num_pairs}-pair subset ==")
    n_pairs = build_subset(
        ROOT / args.source, ROOT / args.subset, args.num_pairs, ["de", "en"]
    )

    # the model dir is read from the config; keep it simple and derive it
    model_dir_line = [
        ln for ln in config.read_text(encoding="utf-8").splitlines()
        if ln.startswith("model_dir:")
    ]
    assert len(model_dir_line) == 1, "could not find model_dir in the config"
    model_dir = ROOT / model_dir_line[0].split(":", 1)[1].strip().strip('"')
    if model_dir.exists() and not args.keep_model_dir:
        shutil.rmtree(model_dir)

    print(f"\n== training on {n_pairs} pairs (train == dev) ==")
    proc = subprocess.run(
        [sys.executable, "-m", "joeynmt", "train",
         str(config)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:], file=sys.stderr)
        print("FAIL: training exited with a non-zero status")
        return 1

    # every pair must actually reach the optimizer. Tokenised length filtering
    # (data.src.max_length) silently drops pairs from training while leaving the
    # dev set intact, which looks exactly like a model that cannot converge.
    seqs = [
        int(m) for m in re.findall(
            r"num\. of seqs:\s*(\d+)", (model_dir / "train.log").read_text("utf-8")
        )
    ]
    if seqs and max(seqs) != n_pairs:
        print(
            f"\nFAIL: only {max(seqs)} of {n_pairs} pairs were trained on. "
            f"Raise data.src.max_length / data.trg.max_length in the config."
        )
        return 1

    validations = model_dir / "validations.txt"
    assert validations.is_file(), f"missing {validations}"
    rows = parse_validations(validations)
    assert rows, "no validation points recorded"

    print(f"\n== loss curve ({len(rows)} validation points) ==")
    print(f"{'steps':>7}  {'loss/token':>10}  {'ppl':>10}  {'bleu':>7}  {'acc':>6}")
    for row in rows:
        print(
            f"{int(row['Steps']):7d}  {row.get('loss_per_token', float('nan')):10.4f}"
            f"  {row.get('ppl', float('nan')):10.4f}"
            f"  {row.get('bleu', float('nan')):7.2f}"
            f"  {row.get('acc', float('nan')):6.3f}"
        )

    best_loss = min(r["loss_per_token"] for r in rows if "loss_per_token" in r)
    best_bleu = max(r["bleu"] for r in rows if "bleu" in r)
    final = rows[-1]

    print("\n== result ==")
    print(f"best loss/token : {best_loss:.4f}   (target < {args.max_loss})")
    print(f"best greedy BLEU: {best_bleu:.2f}   (target > {args.min_bleu})")
    print(f"final step      : {int(final['Steps'])}")

    failures = []
    if not best_loss < args.max_loss:
        failures.append(
            f"loss/token {best_loss:.4f} did not fall below {args.max_loss}: " +
            describe_plateau(rows, "loss_per_token")
        )
    if not best_bleu > args.min_bleu:
        failures.append(
            f"greedy BLEU {best_bleu:.2f} did not exceed {args.min_bleu}: " +
            describe_plateau(rows, "bleu")
        )

    if failures:
        # a vocabulary that cannot represent the references caps BLEU no matter
        # how well the model fits, and shows up as <unk> in the hypotheses
        hyps = sorted(model_dir.glob("*.hyps.dev"))
        if hyps:
            n_unk = hyps[0].read_text(encoding="utf-8").count("<unk>")
            if n_unk:
                failures.append(
                    f"the hypotheses contain {n_unk} <unk> tokens: the vocabulary "
                    f"cannot represent the references, so BLEU is capped "
                    f"independently of the model"
                )
        print("\nFAIL: did not converge")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nPASS: the model memorised the subset")
    return 0


if __name__ == "__main__":
    sys.exit(main())
