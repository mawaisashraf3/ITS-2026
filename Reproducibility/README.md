# Manuscript reproducibility

The study analyzes the VIA miniature physical driving track, with a 256/64/610 train/model-selection/test split. The original image condition and two matched synthetic corruptions were evaluated by four released seed-7 checkpoints. The input evidence is pinned to [the RMLDNet release commit](https://github.com/mawaisashraf3/ITS-2026/commit/f99112fdd8c0f736d96a2661f3940f051b71501d). The repository provides native per-image JSONL, aggregate metrics, split manifests, histories, configurations, model code and three small-model checkpoints.

From the repository root with the documented environment installed, use:

```bash
python Reproducibility/manuscript_analysis.py --root . --output Reproducibility/Outputs
python Reproducibility/verify_extension.py --root . --output Reproducibility/Outputs
python Reproducibility/manuscript_figures.py --root . --analysis Reproducibility/Outputs --output Figures
```

The first script checks the released SQLite snapshot hash and validates all split records, the 7,320 per-image observations, the 12 aggregate confusion matrices, the frozen validation thresholds, and 60 sensitivity cells. It re-derives the pixel metrics and adds matched filename-group bootstrap and leave-one-group-out contrasts. The extension verifier reconciles 9,150 per-image records across 15 conditions, including three exact original-image replays. It also recomputes latency summaries from 360 recorded forwards and aggregates the 40 nominal/stress synthetic verification runs. Extension inputs are pinned to [RMLDNet v0.4.0](https://github.com/mawaisashraf3/ITS-2026/commit/b5094f2dc7ba417ab1a3ecdfc3bf65455ac36dff). The figure script writes four manuscript figures as 400 dpi PNG and vector PDF.

## Files

- `metrics_verified.csv` reports pooled counts, recalculated metrics, parameters, and validation selections.
- `paired_contrasts.csv` reports the paired F1 differences, changes in those differences relative to the original condition, 95% percentile intervals, and the leave-one-group-out ranges. The same 20,000 group draws are used across models and conditions (seed 8603).
- `group_counts.csv` stores all per-group confusion aggregates.
- `model_intervals_verified.csv` stores the independently recomputed model-wise intervals.
- `sensitivity_verified.csv` and `fixed_threshold.csv` contain the checked compact checkpoint sweep and the fixed-threshold precision/recall/F1 series.
- `verification.json` records the input commit, snapshot hash, split sizes, resampling settings, and reconciliation count.
- `extended_fog_verified.csv`, `latency_verified.csv`, and `kalman_summary_verified.csv` contain the checked v0.4.0 diagnostics.

The original study’s model-wise intervals use 2,000 group draws (seed 7). The paired contrasts describe variability under grouping by source filename blocks. Those blocks approximate local dependence and do not establish independent drives or uncertainty across training seeds.

The commands above reanalyze saved experimental observations without downloading images. To regenerate the original predictions and train the published configurations, follow the repository’s [measured-data reproduction procedure](https://github.com/mawaisashraf3/ITS-2026/blob/f99112fdd8c0f736d96a2661f3940f051b71501d/docs/REAL_DATA_EXPERIMENTS.md). The VIA source has semantic masks only. Synthetic corruption coefficients do not quantify measured fog or visibility distance.

## Software checks

All 29 tests from the original evidence release passed on the local CPU environment. A separate two-epoch synthetic fixture run exercised data generation, training, checkpoint saving, and evaluation. Its small evaluation set and all-foreground predictions serve only as a workflow check. Its results are separate from the VIA study.

The v0.4.0 analytic filter suite was rerun locally with `python -m rmldnet.extensions kalman --output runs/manuscript_kalman`. All 40 nominal/stress verification cases agreed with the published metrics within 1e-12. The rerun metrics and comparison record are in `Software_Checks`.

```bash
python -m pytest -q
python -m rmldnet.cli synthetic --output runs/manuscript_smoke_data --count 12
python -m rmldnet.cli train --config configs/smoke.yaml --manifest runs/manuscript_smoke_data/train.jsonl --val-manifest runs/manuscript_smoke_data/val.jsonl --output runs/manuscript_smoke
python -m rmldnet.cli evaluate --config configs/smoke.yaml --checkpoint runs/manuscript_smoke/last.pt --manifest runs/manuscript_smoke_data/val.jsonl --output runs/manuscript_smoke_eval
```
