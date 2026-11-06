# Reproduce the additional computational experiments

The new suite adds six experiment groups and seven figures while preserving all earlier verified result files. It uses the existing trained checkpoints and fixed thresholds, not newly tuned test-set models.

## Prerequisites

Install the existing pinned Python environment and prepare the VIA data with `python -m rmldnet.public_data --download --output data/via`. The compact and U-Net weights are delivered in `results/via`. To include full Swin-B in the controlled efficiency comparison, reproduce its trained checkpoint using the existing commands in `docs/REAL_DATA_EXPERIMENTS.md`; then supply its path explicitly.

```bash
python -m rmldnet.extensions all --output runs/extensions --full-checkpoint runs/via/swin_b/best.pt
```

The `all` command runs isolated efficiency first, before other computation, then synthetic filtering, controlled-image integration, held-out diagnostics, database ingestion, figures and a report. Individual stages are also executable:

```bash
python -m rmldnet.extensions efficiency --output runs/extensions --full-checkpoint runs/via/swin_b/best.pt
python -m rmldnet.extensions kalman --output runs/extensions
python -m rmldnet.extensions robustness --output runs/extensions
python -m rmldnet.extensions figures --output runs/extensions
```

Do not run other CPU-heavy jobs concurrently with the efficiency stage. It launches a fresh subprocess per model, with two Torch threads, batch 1, five warmups and 3×30 timed forwards. Raw timings are retained; sampled RSS is total resident process memory, not allocator or GPU peak. Data loading, image geometry and filtering are outside the forward timing window.

## Data and selection boundaries

The 610 held-out VIA images are independent examples for these analyses, not asserted consecutive video frames. New fog levels are β=0.15, 0.45, 1.8 and 2.1; original images are replayed to verify historical confusion matrices and to support new confidence/geometry diagnostics. Atmospheric depth and airlight are unchanged. Three frozen models are compared: compact with restoration, compact without restoration and U-Net.

The threshold/fog sensitivity study from the earlier release is preserved and not duplicated here. Thresholds remain validation-selected. No calibration temperatures, geometry anchors, model weights or thresholds are fit to held-out labels. New calibration statistics describe the existing predictions. Reference-mask geometry is only used for scoring the same extraction rule, never to fill missing predicted measurements.

## Results and artifacts

- `kalman/`: synthetic nominal/stress traces, independent-noise metrics, the preserved failed initialization trace and corrected verification records.
- `temporal/`: 48 explicitly controlled synthetic images and the trained segmentation → geometry → filter trace. These are generated fixtures, not real driving footage.
- `robustness/`: compatible per-model evaluation bundles, raw per-image counts, calibration bins, geometry records and derived mask/probability examples.
- `efficiency/`: actual per-forward latencies, model sizes, checkpoints and sampled process memory.
- `figures/`: seven PNG/SVG/PDF figure sets from these computed records.
- `experiments.sqlite3`: separate working registry containing the original and additional segmentation evaluations. Filtering metrics retain their own JSON/CSV schema rather than being forced into a pixel-confusion table.

The original public registry remains unchanged. Repeated ingestion is idempotent. The new binary-segmentation bundles can also be imported using the existing database CLI.

## Scientific limitations

The synthetic Monte Carlo experiments vary observation noise and missing/outlier masks on a fixed analytic trajectory. They do not establish generalization to unseen roads or dynamics. The recovery-only rule was refined after a failed development seed. A further initialization challenge exposed startup weakness; stable-window initialization was then fixed before final verification seeds 207–226 were evaluated. Both prior failure sets are retained. Noise covariances are specified for simulation and not calibrated to actual camera measurements.

Geometry coverage is limited: both boundaries often do not bracket the image center in the selected row bands. Conditional MAE must be interpreted together with availability counts and the 610-image denominator. Calibration averages are dominated by background; lane/background Brier errors are reported separately. Entropy is predictive rather than epistemic uncertainty. All model comparisons retain their different original training budgets and initialization, and no causal restoration or statistical-superiority claim is inferred from a single trained checkpoint per architecture.

## Query the added segmentation evaluations

The separate registry contains seven runs: four original runs and three frozen-checkpoint evaluation extensions. Baseline β=0 is a replay, not an independent new test set. Use the run key/configuration to distinguish suites:

```sql
SELECT m.run_key,m.model,m.beta,m.f1
FROM metrics m JOIN experiments e ON e.run_key=m.run_key
WHERE json_extract(e.config_json,'$.evaluation_extension') IS NOT NULL
ORDER BY m.model,m.beta;
```
