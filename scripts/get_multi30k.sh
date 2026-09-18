#!/usr/bin/env bash
#
# Download and preprocess Multi30k de-en (WMT16 multimodal task 1).
#
# Usage:
#   $ bash scripts/get_multi30k.sh              # call from the repo root
#   $ MERGES=4000 bash scripts/get_multi30k.sh  # fewer BPE merges
#
# Produces, in test/data/multi30k/:
#   train.de train.en            29000 raw pairs   <- config `data.train`
#   dev.de   dev.en               1014 raw pairs   <- config `data.dev`
#   test.de  test.en              1000 raw pairs   <- config `data.test`
#   bpe.${MERGES}.codes                           <- config `tokenizer_cfg.codes`
#   vocab.txt                                     <- config `voc_file` (both sides)
#   train.bpe.${MERGES}.{de,en}                   segmented train, used for the vocab
#
# The BPE codes and the vocabulary are learned from the TRAINING SPLIT ONLY.
# Nothing from dev or test touches them. Reusing a vocabulary built on other
# data caps BLEU through <unk> no matter how good the model is: the overfit
# test hit exactly that with test/data/toy/bpe200.txt (see REPORT_NOTES.md).
#
# Source: https://github.com/multi30k/dataset (the official release). The raw
# files are untokenized and cased.
#
# IMPORTANT: the config must not transform the text before BPE differently from
# how the codes were learned here. Keep, in both `data.src` and `data.trg`:
#     lowercase: False
#     normalize: False
#     tokenizer_cfg: {pretokenizer: "none"}
# If you want lowercasing or Moses pretokenization, apply it to the files in
# this script and relearn the codes, so that training and vocabulary agree.

set -euo pipefail

MERGES="${MERGES:-10000}"     # joint BPE merge operations
MIN_FREQ="${MIN_FREQ:-1}"     # drop BPE tokens rarer than this from the vocab
DATA_DIR="${DATA_DIR:-test/data/multi30k}"
BASE_URL="https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw"

# upstream name -> our split name, and the line count we expect
SPLITS=("train:train:29000" "val:dev:1014" "test_2016_flickr:test:1000")
LANGS=(de en)

if [ ! -d "joeynmt" ]; then
    echo "ERROR: run this from the repository root (no joeynmt/ directory here)." >&2
    exit 1
fi

mkdir -p "${DATA_DIR}"

############################################################
echo "== 1/5 downloading =="
############################################################
for entry in "${SPLITS[@]}"; do
    upstream="${entry%%:*}"
    for lang in "${LANGS[@]}"; do
        out="${DATA_DIR}/${upstream}.${lang}"
        if [ -s "${out}" ]; then
            echo "  ${out} exists, skipping"
            continue
        fi
        echo "  ${BASE_URL}/${upstream}.${lang}.gz"
        curl -fL --retry 3 --retry-delay 2 -o "${out}.gz" \
            "${BASE_URL}/${upstream}.${lang}.gz"
        gunzip -f "${out}.gz"
    done
done

############################################################
echo "== 2/5 laying out splits and verifying line counts =="
############################################################
fail=0
for entry in "${SPLITS[@]}"; do
    upstream="${entry%%:*}"
    rest="${entry#*:}"
    split="${rest%%:*}"
    expected="${rest##*:}"

    for lang in "${LANGS[@]}"; do
        src="${DATA_DIR}/${upstream}.${lang}"
        dst="${DATA_DIR}/${split}.${lang}"
        [ "${src}" != "${dst}" ] && mv -f "${src}" "${dst}"

        actual=$(wc -l < "${dst}" | tr -d ' ')
        if [ "${actual}" != "${expected}" ]; then
            echo "  ERROR: ${dst} has ${actual} lines, expected ${expected}" >&2
            fail=1
        else
            echo "  ${dst}: ${actual} lines"
        fi
    done

    # the two sides must line up pair for pair
    n_de=$(wc -l < "${DATA_DIR}/${split}.de" | tr -d ' ')
    n_en=$(wc -l < "${DATA_DIR}/${split}.en" | tr -d ' ')
    if [ "${n_de}" != "${n_en}" ]; then
        echo "  ERROR: ${split} is not parallel: ${n_de} de vs ${n_en} en" >&2
        fail=1
    fi
done
if [ "${fail}" != "0" ]; then
    echo "Aborting: the download does not match the expected Multi30k sizes." >&2
    exit 1
fi

############################################################
echo "== 3/5 learning joint BPE on the training split only (${MERGES} merges) =="
############################################################
CODES="${DATA_DIR}/bpe.${MERGES}.codes"
TRAIN_JOINT="${DATA_DIR}/train.joint.tmp"

cat "${DATA_DIR}/train.de" "${DATA_DIR}/train.en" > "${TRAIN_JOINT}"
python -m subword_nmt.learn_bpe \
    --input "${TRAIN_JOINT}" \
    --output "${CODES}" \
    --symbols "${MERGES}" \
    --min-frequency "${MIN_FREQ}"
rm -f "${TRAIN_JOINT}"
echo "  wrote ${CODES}"

############################################################
echo "== 4/5 building the joint vocabulary from the segmented training split =="
############################################################
for lang in "${LANGS[@]}"; do
    python -m subword_nmt.apply_bpe \
        --codes "${CODES}" \
        --input "${DATA_DIR}/train.${lang}" \
        --output "${DATA_DIR}/train.bpe.${MERGES}.${lang}"
done

VOCAB="${DATA_DIR}/vocab.txt"
cat "${DATA_DIR}/train.bpe.${MERGES}.de" "${DATA_DIR}/train.bpe.${MERGES}.en" \
    | python -m subword_nmt.get_vocab \
    | awk -v min="${MIN_FREQ}" '$2 >= min {print $1}' > "${VOCAB}"

# JoeyNMT's Vocabulary prepends <unk> <pad> <s> </s> itself, so the file holds
# only real tokens (joeynmt/vocabulary.py, Vocabulary.__init__).
echo "  wrote ${VOCAB} ($(wc -l < "${VOCAB}" | tr -d ' ') types)"

############################################################
echo "== 5/5 coverage check: how much of each split the vocabulary can represent =="
############################################################
# An OOV subword becomes <unk>, which caps BLEU independently of the model.
# Train should be 0.00%; dev/test should be small. If dev/test are high, raise
# MERGES or check that the config does not transform the text differently.
python - "${DATA_DIR}" "${MERGES}" "${VOCAB}" <<'PY'
import sys
from pathlib import Path

from subword_nmt import apply_bpe

data_dir, merges, vocab_file = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
with (data_dir / f"bpe.{merges}.codes").open(encoding="utf-8") as codes:
    bpe = apply_bpe.BPE(codes)
vocab = set(vocab_file.read_text(encoding="utf-8").split("\n")) - {""}

print(f"  {'split':<12}{'tokens':>10}{'OOV':>8}{'OOV %':>8}")
for split in ("train", "dev", "test"):
    for lang in ("de", "en"):
        total = oov = 0
        path = data_dir / f"{split}.{lang}"
        for line in path.read_text(encoding="utf-8").splitlines():
            for token in bpe.process_line(line.strip()).split():
                total += 1
                oov += token not in vocab
        pct = 100.0 * oov / total if total else 0.0
        print(f"  {split + '.' + lang:<12}{total:>10}{oov:>8}{pct:>7.2f}%")
PY

cat <<EOF

== done ==
Point the config at these files:

    data:
        train: "${DATA_DIR}/train"
        dev:   "${DATA_DIR}/dev"
        test:  "${DATA_DIR}/test"
        dataset_type: "plain"
        src: &side
            level: "bpe"
            lowercase: False
            normalize: False
            voc_min_freq: ${MIN_FREQ}
            voc_file: "${VOCAB}"
            tokenizer_type: "subword-nmt"
            tokenizer_cfg:
                num_merges: ${MERGES}
                codes: "${CODES}"
                pretokenizer: "none"

Equivalent config-driven route once such a config exists:
    python scripts/build_vocab.py <config.yaml> --joint
It rebuilds ${CODES} and ${VOCAB} from the config's training split alone.
EOF
