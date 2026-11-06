# Reproduce the measured-data experiment

The actual data used here are VIA / GGDataSet camera images of a **miniature physical driving track**, distributed with semantic masks for road, line and background. These are additional relevant lane data, not the TuSimple, CULane, BDD100K or FoggyLane benchmarks. Original images remain outside Git.

## Obtain and prepare

From the repository root with the environment activated:

```bash
python -m rmldnet.public_data --download --output data/via
```

This downloads a 1,619,347,448-byte public GitHub release, verifies the pinned SHA-256 before processing, and extracts only image/mask data. Executable scripts, pretrained `.h5`/`.pb` models and Mac metadata bundled in the archive are ignored. The SHA-256 is recorded in code and provenance. The repository license is MIT (Maker Hanoi, 2021); see `THIRD_PARTY_NOTICES.md`. If download fails, provide the exact archive via `--archive /path/to/Segmentation_Dataset.zip` rather than bypassing TLS/checksum checks.

The release has 5,074 original training frame files and 1,157 original validation frame files. Exact-byte duplicates within and across the original splits are removed before selection. The remaining original validation set is reserved as the held-out test set. Training and model-selection validation use disjoint 100-index filename blocks of the original training set. Hash ranking with seed 7 selects 256 training frames and 64 validation frames. This block scheme reduces immediate neighboring-frame leakage but cannot prove independent sequences because the release lacks video IDs. All split hashes/membership and duplicate records are published in the results directory.

The mask color `(192,128,128)` is lane/line, all other documented colors are background for this binary task. Unknown colors fail preparation. Every record sets `instance_supervision=false`; instance embeddings are not trained using invented instance identities. This dataset experiment tests semantic lane detection and restoration, not lane-instance separation or temporal tracking.

## Train and evaluate

Download the official, hash-verified Swin-B ImageNet-1K checkpoint:

```bash
python - <<'PY'
import torch
from torchvision.models import Swin_B_Weights
w = Swin_B_Weights.IMAGENET1K_V1
torch.hub.load_state_dict_from_url(w.url, model_dir='data/weights', check_hash=True, map_location='cpu')
PY
python -m rmldnet.experiment --model swin_b --encoder data/weights/swin_b-68c6b09e.pth --epochs 3 --output runs/via/swin_b
python -m rmldnet.experiment --model compact --epochs 20 --output runs/via/compact
python -m rmldnet.experiment --model unet --epochs 20 --output runs/via/unet
python -m rmldnet.experiment --model compact_no_defog --epochs 20 --output runs/via/compact_no_defog
python -m rmldnet.report --runs runs/via/swin_b runs/via/compact runs/via/unet runs/via/compact_no_defog --output results/via
```

CPU runs use two PyTorch threads per process, batch 4 and 96×224 inputs. Full Swin-B retains its standard width/depths and optimizes **all** parameters, with ImageNet-1K initialization. Its encoder LR is 1e-5 and new-module LR is 1e-3; compact/U-Net use 1e-3, AdamW weight decay .01 and cosine scheduling. The restoration-enabled variants first receive two epochs of paired atmospheric synthetic-fog reconstruction on the training images, followed by joint lane-supervised optimization. The compact Swin width is 16, depths [2,2,2,2], decoder width 32; it is an explicitly smaller implementation variant. U-Net width is 16. These are staged CPU training budgets at reduced input resolution.

Every epoch evaluates only model-selection validation. The best validation pixel F1 chooses the checkpoint; a fixed threshold grid 0.10..0.90 by 0.05 is searched only on validation predictions. Ties prefer a threshold nearer 0.5. The checkpoint and threshold are frozen before the held-out test is read. Final test evaluates all deduplicated held-out images three times: original, synthetic atmospheric fog beta 0.7, beta 1.2; airlight 0.85 and normalized vertical depth 0.4..1.8. These are repeatable corruptions, not actual measured fog.

Reported P/R/F1/accuracy/IoU/mIoU aggregate the entire pixel confusion matrix. Per-image metrics, raw confusion matrices, actual elapsed evaluation times, epoch histories, configs and trained-checkpoint tensor hashes are saved. Timings include loading and forward passes and must not be labeled model-only FPS. No result from literature is mixed into measured tables. This single-seed study is descriptive; it does not establish statistical significance or generalization to full-size roads.

## Re-evaluate published small-model checkpoints

The results include `best.pt` for compact/U-Net variants (original trained code weights only):

```bash
python -m rmldnet.experiment --model compact --output results/via/compact --evaluate-only
```

Use the matching `--model` for the other checkpoints. The 350+ MB full Swin-B checkpoint is retained locally under `runs/via/swin_b/best.pt` and is excluded from ordinary Git; reproduce it by training with the command above. Deterministic algorithms and a fixed seed are used, but bitwise equivalence across PyTorch/platform versions is not promised. Other benchmark targets require their own releases, licenses, split manifests and evaluators. Official TuSimple's documented S3 host is presently blocked by network policy; a narrowly scoped allowlist addition was saved for user review.

## Dataset access for the road benchmarks

The official TuSimple wiki links `https://tusimple-benchmark-evaluation.s3.amazonaws.com/datasets/1/train_set.zip`. The configured proxy returned `403 Forbidden` while establishing the connection to this host. A supported environment-draft update added only that hostname; saving the draft does not itself change the running policy. No blocked endpoint was accessed through a mirror or alternate proxy. Public additional datasets hosted on already-allowed GitHub release domains were used instead. No login credentials or token values are needed for the VIA release.

## Inference with a delivered checkpoint

```bash
python -m rmldnet.predict --experiment results/via/compact --images /path/to/frame1.png /path/to/frame2.jpg --output runs/new_images
```

This loads `config.json` plus `best.pt`, applies the validation-selected threshold, and writes masks at original image resolution, network-resolution probabilities, green overlays and a predictions manifest. Use `results/via/unet` or `results/via/compact_no_defog` to use another delivered model. These semantic checkpoints have no instance supervision and should not be used to claim lane-instance or tracking performance; the instance-supervised pipeline supports instance-supervised datasets and tracking.
