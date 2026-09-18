# Experiments

One row per training run. Columns as specified in CLAUDE.md. The BLEU column
carries the best greedy dev score (what early stopping tracks) followed by the
beam-5 dev and test scores from the final evaluation.

All three Multi30k runs share the same data, tokenizer, vocabulary, beam size 5,
length penalty 1.0, sacrebleu `tokenize: "13a"` and training schedule; only the
`model:` block differs. See REPORT_NOTES.md.

| date | git commit | config | model | params | steps | best dev BLEU | wall-clock | throughput | hardware | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-18 | d0c7a68 | multi30k_conv.yaml | ConvS2S | 10,983,936 | 24,062 steps / 100 epochs | 29.85 greedy (step 23000); beam 5: dev 32.87, test 34.16 | 32.4 min | ~30k tok/s | Colab T4, fp16 | epoch cap reached, no early stop, LR never annealed from 3e-4 |
| 2026-09-18 | 550e2ca | multi30k_transformer.yaml | Transformer 3enc/2dec | 10,862,336 | 24,062 steps / 100 epochs | 31.87 greedy (step 24000); beam 5: dev 34.28, test 35.06 | 33.9 min | ~26k tok/s | Colab T4, fp16 | epoch cap reached, dev BLEU still rising at final validation |
| 2026-09-18 | 550e2ca train / b5f54a9 decode | multi30k_rnn.yaml | biGRU 2enc/2dec + Luong | 11,116,032 | 24,062 steps / 100 epochs | 14.49 greedy (step 24000); beam 5: dev 16.02, test 16.66 | 72.7 min (+ separate re-decode) | ~10.2k tok/s | Colab T4, fp16 | beam decode crashed on the fp16 `_init_hidden` bug; re-decoded from the checkpoint after the fix. Clearly undertrained at this budget |

Notes on this table:
- `throughput` is an extra column beyond the schema in CLAUDE.md, added once
  tokens/sec became one of the comparison's findings.
- Commit hashes for the transformer and RNN rows were not recorded with the
  runs and are inferred: `550e2ca` is the commit that fixed the training
  schedule, and the RNN's beam decode necessarily post-dates the fp16 fix in
  `b5f54a9`. Correct them if the Colab notebooks say otherwise.
