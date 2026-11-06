# Executed validation

## Additional measured-data validation

The repository now also includes an actual VIA / GGDataSet experiment, documented in [REAL_DATA_EXPERIMENTS.md](REAL_DATA_EXPERIMENTS.md) and [results/via](../results/via/README.md). Training and held-out evaluation use real camera images from a miniature physical track. The split contains 256 training, 64 validation and 610 held-out test images after exact duplicate removal. Instance supervision is disabled because the source provides semantic classes only.

The expanded suite has **29 passing tests**, including semantic-supervision masking, U-Net gradients, validation threshold selection, measured confusion matrices, archive checksum rejection, duplicate removal, mask conversion and delivered-checkpoint inference. A separately loaded compact checkpoint exactly reproduced all three held-out confusion matrices and scalar metrics. No real source images are committed. The following section preserves the earlier synthetic/software validation separately.

## Earlier software and synthetic validation

Current-instance validation performed on CPU, Python 3.12.14, torch 2.5.1+cpu / torchvision 0.20.1+cpu. `requirements.txt` pins direct dependencies; `requirements-cpu-lock.txt` records the full tested dependency set. For the latter, install the CPU PyTorch wheels as in the README first, then `pip install -r requirements-cpu-lock.txt`. No CUDA device is available.

## Passed

- `python -m pytest -q`: **11 tests passed**, including full composite-loss backward through restoration/encoder/both heads, odd image sizes, known confusion-matrix results, empty-instance loss, label permutation, atmospheric inversion, polyline geometry, clustering, gated association, track aging/reset, covariance positive definiteness, stationary synthetic jitter reduction, all three annotation converters, encoder loading and disabled-module paths.
- `pip check`: no broken dependencies.
- Synthetic dataset generation: 12 independent generated scenes, seed 7; 9 training and 3 held-out validation scenes.
- Synthetic restoration pretraining: 2 epochs; mean L1 training loss 0.2632 → 0.2320.
- Full smoke pipeline: 2 epochs, checkpoint write/reload, independent evaluation, two-image inference output and 20-step network latency measurement completed.
- Full Swin-B with restoration/FPN/head: finite forward output for input `[1,3,320,800]`, logits `[1,1,320,800]`, embeddings `[1,4,320,800]`, 87,079,889 parameters. Random initialization; CPU forward about 1.92 seconds in this single check. Not a steady-state FPS benchmark.
- Ablation runner executed full/no-defog/no-FPN variants, 2 epochs each, seed 7. These only validate executable branches, not comparative research conclusions.
- Extended synthetic optimization check: 40 epochs, LR 1e-3 (optimization check), seed 7, same 9/3 split, pretrained synthetic restoration initialization. Mean training loss 6.5200 → 1.5662; held-out pixel F1 0.3156 at final epoch. This demonstrates optimization, not meaningful real-road performance.

## Interpretation

The two-epoch smoke run predicts all pixels as lane: P=0.0651, R=1, F1=0.1222. This intentionally short run does **not** establish detector quality. Its CPU network-only speed (~126.6 FPS at **128×64** on the 587,255-parameter smoke model) is not comparable to full Swin-B. Evidence JSON contains actual measured values and explicit measurement scope.

The extended run remains weak even on simple synthetic data. Longer, properly initialized training and real held-out evaluation are necessary before any research accuracy claim. All reported experiment measurements were produced by executed runs.

## Not executed / unresolved

- TuSimple, CULane, BDD100K and exact FoggyLane benchmark training or official evaluation: licensed datasets and exact split manifests were not supplied.
- ImageNet-22K initialization: not evaluated.
- Four-A100 distributed training, CUDA numerical behavior and real-time end-to-end performance: unavailable; this trainer uses one selected device.
- Actual fog-domain adaptation/generalization, real-video temporal improvements, baseline comparisons and statistical significance: require real datasets, temporal order, validated protocols and multiple seeds.
- Automatic adaptation under scene cuts or camera calibration changes: not implemented; explicitly reset tracking at sequence boundaries.

Evidence files exclude weights, copyrighted images. Full local checkpoints/data are ignored under `runs/` and `data/`.

## Platform and database verification

The current suite additionally verifies atomic result ingestion, idempotence, SQL-derived metrics, foreign keys, integer-count constraints, read-only query authorizers, insert/update split guards, invalid-input rollback, online backup/recovery, JSON export/import, configuration validation and full management-pipeline execution. `results/platform_verification.json` records the latest executed suite and package checks. `results/database/verification.json` records measured database operation timings and the delivered registry checksum.

The analysis workflow completed 2,000 source-group bootstrap resamples and a six-level atmospheric-corruption sweep across ten thresholds using the fixed compact checkpoint. Original and beta=1.2 confusion matrices exactly matched the previously evaluated model. The paired F1-difference interval includes zero; no reliable superiority claim is inferred.

## v0.4 filtering and diagnostic extension

The new release has 47 passing tests, including scalar and correlated-state hand calculations, covariance stability, missing/partial observations, rejected innovations, stable initialization, lost-track recovery, sequence provenance and controlled-image integration. The complete new experiment command was executed from a fresh output directory, including reproduction of both failed initialization variants. Original segmentation confusion matrices replayed exactly for all three delivered checkpoints. All earlier result artifacts remain unchanged.

See `results/extensions/verification.json` for the current test, packaging, artifact-preservation and database checks; the preceding platform evidence describes v0.3. Numerical trajectories and rendered sequences are explicitly synthetic. Real-image geometry is coverage-limited and is not vehicle localization ground truth.
