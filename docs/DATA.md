# Data preparation and provenance

Obtain licensed data yourself from the respective official projects:

- TuSimple: https://github.com/TuSimple/tusimple-benchmark
- CULane: https://xingangpan.github.io/projects/CULane.html
- BDD100K: https://www.bdd100k.com/ (use lane `poly2d` labels matching the downloaded image release).
- FoggyLane: exact release and split must be independently verified; no guessed download URL or synthetic substitute is called FoggyLane.

Converters never download or redistribute datasets. Record dataset version, license, original dimensions, split IDs and SHA256 checksums in your experiment records. Preserve held-out test sets. Where no official validation split is supplied, select and record a sequence-disjoint split; the split selection must be recorded.

```bash
python -m rmldnet.prepare culane --annotations /datasets/CULane/list/train_gt.txt --images /datasets/CULane --output data/culane/train
python -m rmldnet.prepare bdd100k --annotations /datasets/bdd/labels/lane_train.json --images /datasets/bdd/images/train --output data/bdd/train
```

Repeat for each existing split. Width defaults to 5 **original image pixels**, a modeling assumption, not the official CULane evaluation width. Masks are resized by nearest-neighbor; images by bilinear interpolation. ID 0 is background; 1..65535 are separate lanes. Overlapping lane strokes use last-drawn ID. BDD cubic control points are sampled into polylines; inspect conversion overlays against your dataset version before research training. Labels with category starting `lane` are included; determine whether curbs/road edges should be excluded in your protocol.

FoggyLane or custom masks use JSONL directly:

```json
{"image":"images/frame.png","mask":"masks/frame.png","sequence":"drive-01","frame":12}
```

Paths are relative to the manifest directory (absolute paths also work). Each grayscale uint16 PNG mask must align with the source image. A binary mask treats all lane pixels as one instance; it does **not** supply instance supervision. Every positive value must represent a real separate lane. Sequence metadata are provenance only; the current training loader is frame-based and pixel evaluation does not run tracking. For tracking, pass explicitly ordered frame paths to `infer --track`, one sequence per invocation.

Synthetic generation creates simple curved lane fixtures, seed 7 by default, with independent train/validation scenes and a provenance marker. Atmospheric augmentation uses an uncalibrated vertical depth gradient, beta U(.2,1.5) and airlight U(.7,1). It is a robustness augmentation, not a physically measured reconstruction of real fog. Restoration pretraining uses clean augmented inputs as targets and synthesized fog on a fraction of samples; original real-fog images must not be used as paired clear ground truth.
