# Report notes

Running log of everything the final report will need.
Append as work happens. Do not reconstruct at the end.

## Design decisions and why

## Paper ambiguities and resolutions

## Correctness evidence

## Bugs and fixes

## Environment and reproducibility
- Local: conda env "convs2s", Python 3.11.16, torch 2.1.2, CPU only
- Upstream baseline commit: cdc4d03d430a1b0f29793a0d95743c5e72ae2f6c (2024-01-25)
- Pins required to run JoeyNMT v2.3 in 2026:
  numpy<2 (torch 2.1.2 NumPy 1.x ABI), sentencepiece==0.1.99
  (0.2.x drops SetVocabulary), importlib_metadata (missing from requirements)
- Baseline test suite: OK (skipped=1)
- Colab setup: TBD

## Results and observations

## Limitations and open questions

## References
- Gehring et al. 2017, Convolutional Sequence to Sequence Learning
- Dauphin et al. 2017, Language Modeling with Gated Convolutional Networks
- Kreutzer et al. 2019, Joey NMT: A Minimalist NMT Toolkit for Novices
