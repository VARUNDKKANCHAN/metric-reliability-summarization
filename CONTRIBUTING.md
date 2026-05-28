# Contributing

Thank you for your interest in contributing!

## Reporting Issues

Please open a GitHub Issue with:
- Your OS, Python version, GPU/CUDA version
- The exact command you ran
- The full error message / traceback

## Adding a New Metric

1. Copy `scripts/01_bleu_pipeline.py` as a template
2. Replace the metric computation function
3. Keep the same output file structure:
   - `outputs/<metric>_outputs/<metric>_stability_sample_level.csv`
   - `outputs/<metric>_outputs/<metric>_per_example_across_seeds.csv`
   - `outputs/<metric>_outputs/<metric>_corpus_per_seed.csv`
   - `outputs/<metric>_outputs/summ_eval_with_<metric>.csv`
   - `outputs/<metric>_outputs/<metric>_stability_summary.json`
4. Add the metric entry to the `metrics` dict in `scripts/09_unified_analysis.py`
5. Open a pull request

## Code Style

- Use `black` for formatting: `black scripts/`
- Use `isort` for imports: `isort scripts/`
- Keep `FAST_MODE` and `SMOKE_TEST` flags at the top of every pipeline script
