# Project: ConvSeq2Seq in JoeyNMT

## What this is
Solo Proseminar project for "Introduction to Neural Networks".
Goal: implement Gehring et al. (2017) Convolutional Sequence to Sequence
as a new encoder/decoder pair inside JoeyNMT v2.3, then compare it against
JoeyNMT's built-in RNN and Transformer baselines on Multi30k de-en.

Graded coursework. The deliverables are working code, a results table,
and a written report.

## Core design rule
ADD, never REPLACE. ConvEncoder and ConvDecoder are new classes selectable
via config (encoder.type: conv). Existing RNN and Transformer code must
remain untouched and runnable so baselines stay reproducible.

## How to work with me
- Just implement. Don't explain the approach first, don't ask for
  confirmation, don't teach. Write the code.
- Work in whole components. Finish it, run the tests, report pass/fail
  and what changed. That's the whole report back to me.
- Write the unit test alongside every component and run it before
  handing back. Don't ask whether I want tests.
- Keep going until the component works. Don't stop at the first error
  to check in with me.
- If a paper detail is genuinely ambiguous, pick the most standard
  interpretation, implement it, and note the choice in one line.
- Never edit files under joeynmt/ that I have not explicitly named.
- Never modify existing baseline configs or existing tests.

## Architecture details to respect (Gehring et al. 2017)
- GLU: conv maps d -> 2d, split, A * sigmoid(B)
- Encoder conv: pad (k-1)/2 both sides, length preserved
- Decoder conv: left-pad (k-1), then trim (k-1) from the end (causality)
- Residual sums scaled by sqrt(0.5)
- Multi-step attention: EVERY decoder layer attends
- Attention query: d = W*h + b + g   (g = previous target embedding)
- Attention context: c = sum_j a_ij * (z_j + e_j)   <- note the + e_j
- Init: N(0, sqrt(4p/n)) into GLU layers, N(0, sqrt(1/n)) otherwise
- Encoder gradients scaled by 1/num_attention_layers
- Mask padded positions to zero before every convolution

## Environment (do not change without asking me)
- conda env "convs2s", Python 3.11.16
- torch 2.1.2, JoeyNMT 2.3.0 installed with pip install -e .
- Baseline upstream commit: cdc4d03d430a1b0f29793a0d95743c5e72ae2f6c (2024-01-25)
- Pinned deviations from JoeyNMT's requirements, needed to run in 2026:
  * numpy<2                (torch 2.1.2 built against the NumPy 1.x ABI)
  * sentencepiece==0.1.99  (0.2.x drops SetVocabulary from the Python API)
  * importlib_metadata     (imported by helpers.py, not in requirements)
- Full state in requirements-frozen.txt
- Tests: python -m unittest  -> must stay at OK (skipped=1)
- Style: yapf + isort + flake8 (make check)

## Hardware and workflow
Two machines. GitHub is the single source of truth.

Local: MacBook Air (Apple Silicon). CPU only -- JoeyNMT v2.3 has no MPS
support and we are NOT adding any (out of scope, incomplete op coverage).
Local work is small only: unit tests, toy task, overfit-50 tests, debugging.

Training: Google Colab (T4 GPU), possibly a university cluster later.
Colab is disposable -- it clones this repo, installs, trains, pushes
results back. NEVER edit source in Colab. All code changes happen locally.
Colab needs its own dependency pinning (different base torch/python);
record whatever works in REPORT_NOTES.md.

Anything that produces a number for the results table runs on GPU.
Anything that checks correctness runs locally.

## Logging -- two files, both mandatory
EXPERIMENTS.md -- one row per training run:
date | git commit | config | model | params | steps | best dev BLEU |
wall-clock | hardware | notes

REPORT_NOTES.md -- anything the final report will need. Append as we go,
never at the end. Sections:
- Design decisions and why (incl. deviations from the paper)
- Ambiguities in the paper and how they were resolved
- Correctness evidence (what each test proves)
- Bugs hit and what fixed them
- Environment and reproducibility notes (pins, Colab setup, seeds)
- Results, observations, surprises
- Open questions and limitations
- References used

After any non-trivial change, add the relevant lines to REPORT_NOTES.md
without being asked.

After any environment fix, upstream incompatibility, version pin, or
non-obvious design decision, append it to REPORT_NOTES.md in the same
turn, without being asked. Include the actual error message, version
numbers, and measured values where relevant.
