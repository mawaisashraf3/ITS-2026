# Changelog

## 0.4.0

- Added a reusable validated linear Kalman kernel, two-anchor lane-position dynamics, stable initialization, innovation gating and explicit lost-track reacquisition.
- Integrated declared ordered images, trained segmentation, corridor measurements and causal filtering; verified controlled synthetic renders without inferring temporal order from VIA filenames.
- Added six experimental groups and seven PNG/SVG/PDF figure sets for synthetic trajectories, stress, new fog levels, CPU efficiency, confidence/errors and geometry.
- Retained failed initialization experiments, documented their diagnoses, and evaluated final settings on separate verification noise seeds.
- Replayed original confusion matrices, preserved earlier result artifacts, and added compatible segmentation evaluations to a separate registry.

## 0.3.0

- Added an eight-table SQLite experiment registry with transactional ingestion, read-only queries, integrity checks, portable exports and online backup/recovery.
- Added group-bootstrap uncertainty, paired comparisons, and actual threshold/fog sensitivity analysis.
- Added exportable PNG/SVG/PDF figures, platform configuration, orchestration and database integration tests.
- Expanded architecture, algorithm, data-management, recovery and contributor documentation.
- Preserved measured model results and delivered checkpoints with explicit evaluation scope.

## 0.2.0

- Executed VIA miniature-track training for full Swin-B, compact Swin, U-Net and no-restoration variants.
- Published measured pixel metrics, per-image counts, training histories, split manifests and trained compact/U-Net checkpoints.

## 0.1.0

- Implemented restoration, hierarchical encoding, feature pyramids, instance objectives, temporal tracking, converters and numerical tests.
