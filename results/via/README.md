# Measured VIA lane-segmentation results

Pixel metrics, not official lane-instance scores. Original held-out validation images are used as the test set; the training subset and model-selection validation subset come from separate filename blocks of original training data.

| Model | Seed | Synthetic fog beta | Test images | Precision | Recall | F1 | Lane IoU | mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| swin_b | 7 | 0.0 | 610 | 83.18 | 81.51 | 82.34 | 69.98 | 83.91 |
| swin_b | 7 | 0.7 | 610 | 80.59 | 83.81 | 82.17 | 69.74 | 83.74 |
| swin_b | 7 | 1.2 | 610 | 81.26 | 82.07 | 81.66 | 69.01 | 83.36 |
| compact | 7 | 0.0 | 610 | 79.87 | 75.33 | 77.54 | 63.31 | 80.31 |
| compact | 7 | 0.7 | 610 | 81.50 | 74.43 | 77.80 | 63.67 | 80.52 |
| compact | 7 | 1.2 | 610 | 83.04 | 69.04 | 75.39 | 60.51 | 78.87 |
| unet | 7 | 0.0 | 610 | 82.63 | 79.32 | 80.94 | 67.98 | 82.83 |
| unet | 7 | 0.7 | 610 | 81.87 | 81.66 | 81.76 | 69.15 | 83.45 |
| unet | 7 | 1.2 | 610 | 81.79 | 80.28 | 81.03 | 68.10 | 82.89 |
| compact_no_defog | 7 | 0.0 | 610 | 77.68 | 76.45 | 77.06 | 62.68 | 79.93 |
| compact_no_defog | 7 | 0.7 | 610 | 79.13 | 76.72 | 77.90 | 63.81 | 80.56 |
| compact_no_defog | 7 | 1.2 | 610 | 80.52 | 71.74 | 75.87 | 61.13 | 79.16 |

Metrics in percent. Beta 0 uses original images; beta 0.7/1.2 are fixed atmospheric synthetic corruptions applied to the same held-out images. These are not real-fog benchmarks.

The compact defogging ablation does not show a consistent benefit: its synthetic-fog F1 scores are slightly lower with defogging enabled. This staged experiment does not establish a general defogging benefit.

The dataset contains miniature physical-track scenes, not full-size public-road driving. Instance identities are absent: instance loss is disabled and temporal tracking is not scored. TuSimple, CULane, BDD100K and FoggyLane are separate evaluation targets.

All thresholds and best epochs are selected only on validation. Models share the same 96×224 evaluation resolution and split; budgets and initialization differ and are recorded in each config.json. Full Swin-B uses ImageNet-1K pretrained initialization; compact/U-Net start randomly. No superiority or statistical-significance claim follows from a single seed.

Each model directory includes config.json, history.json, metrics.csv, learning.svg and masks.png. Mask columns are ground truth, prediction and error map (green TP, red FN, blue FP); source photographs are excluded.

Small-model checkpoints are provided. The full Swin-B checkpoint exceeds regular Git size limits and is retained in the working environment; regenerate with the documented command. Source photographs and the dataset archive are not included.
