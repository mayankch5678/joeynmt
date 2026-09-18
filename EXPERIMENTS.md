# Experiments

One row per training run. Columns as specified in CLAUDE.md. The BLEU column
carries the best greedy dev score (what early stopping tracks) followed by the
beam-5 dev and test scores from the final evaluation.

All three Multi30k runs share the same data, tokenizer, vocabulary, beam size 5,
length penalty 1.0, sacrebleu `tokenize: "13a"` and training schedule; only the
`model:` block differs. See REPORT_NOTES.md.

| date | git commit | config | model | params | steps | best dev BLEU | wall-clock | hardware | notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-18 | d0c7a68 | multi30k_conv.yaml | ConvS2S | 10,983,936 | 24,062 steps / 100 epochs | 29.85 greedy (step 23000); beam 5: dev 32.87, test 34.16 | 32.4 min | Colab T4, fp16 | epoch cap reached, no early stop, LR never annealed from 3e-4 |
