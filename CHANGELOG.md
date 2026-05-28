# Changelog

All notable changes to this project will be documented here.

## [1.0.0] — 2025

### Added
- Full pipeline for 8 summarization metrics: BLEU, ROUGE-1/2/L, METEOR, chrF++, BERTScore, MoverScore, COMET, BLEURT
- Stochastic stability experiments: 500 CNN/DailyMail documents, 10 seeds, 3 samples/doc (15,000 summaries total)
- Human alignment evaluation on SummEval benchmark (1,600 system–document pairs)
- Unified analysis pipeline: Tables I–II, Figures 1–4, Excel workbook
- Pairwise Mann–Whitney U tests with Holm–Bonferroni correction (28 pairs)
- Bootstrap permutation tests for alignment hierarchy (1,000–10,000 iterations)
- Levene's test for variance homogeneity across seeds
- CV = σ/|mean| support for negative-mean metrics (COMET, BLEURT)
- GitHub Actions smoke test workflow
- Conda environment file
- CITATION.cff for academic citation
