"""Download verified VIA lane data; never execute code bundled in archives."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
import urllib.request
from PIL import Image
import numpy as np

URL = "https://github.com/makerhanoi/via-dataset-cds-2020/releases/download/v1.0/Segmentation_Dataset.zip"
SHA256 = "534448b9c45195c86dc7e0c372372c60e3b00e1a6cd2e7e55d489839aff10ef3"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1024 * 1024):
            h.update(b)
    return h.hexdigest()


def prepare(archive, output, train_count=256, val_count=64, seed=7):
    archive = Path(archive)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if digest(archive) != SHA256:
        raise ValueError("VIA release SHA256 mismatch")
    records = {"train": [], "val": []}
    unknown = set()
    seen = {}
    duplicates = []
    with ZipFile(archive) as z:
        names = set(z.namelist())
        for split in ("train", "val"):
            prefix = f"Segmentation_Dataset/GGDataSet/{split}_frames/{split}/"
            frames = sorted(
                n for n in names if n.startswith(prefix) and n.endswith(".png")
            )
            for name in frames:
                maskname = name.replace(f"{split}_frames", f"{split}_masks")
                if maskname not in names:
                    raise ValueError(f"Missing mask {maskname}")
                image_bytes = z.read(name)
                key = hashlib.sha256(image_bytes).hexdigest()
                # Exact-byte duplicate removal including original train/val overlap.
                if key in seen:
                    duplicates.append({"file": name, "duplicate_of": seen[key]})
                    continue
                seen[key] = name
                from io import BytesIO

                im = Image.open(BytesIO(image_bytes)).convert("RGB")
                lab = Image.open(BytesIO(z.read(maskname))).convert("RGB")
                rgb = np.asarray(lab)
                allowed = (
                    np.all(rgb == [0, 0, 0], axis=-1)
                    | np.all(rgb == [128, 0, 0], axis=-1)
                    | np.all(rgb == [192, 128, 128], axis=-1)
                )
                if not allowed.all():
                    unknown.update(
                        tuple(c.tolist()) for c in np.unique(rgb[~allowed], axis=0)
                    )
                binary = np.all(rgb == [192, 128, 128], axis=-1).astype("uint8")
                if im.size != lab.size:
                    raise ValueError("Source image/mask mismatch")
                stem = Path(name).stem
                image_path = output / "images" / f"{stem}.png"
                mask_path = output / "masks" / f"{stem}.png"
                image_path.parent.mkdir(exist_ok=True)
                mask_path.parent.mkdir(exist_ok=True)
                image_path.write_bytes(image_bytes)
                Image.fromarray(binary).save(mask_path)
                # Group adjacent source indices into blocks as a cautious temporal proxy.
                index = int(stem.split("_")[-1])
                group = f"{split}-{index//100:04d}"
                records[split].append(
                    {
                        "id": stem,
                        "image": f"images/{stem}.png",
                        "mask": f"masks/{stem}.png",
                        "group": group,
                        "source_split": split,
                        "sha256": key,
                        "instance_supervision": False,
                    }
                )
    if unknown:
        raise ValueError(f"Unexpected mask colors: {unknown}")
    groups = sorted(
        {r["group"] for r in records["train"]},
        key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest(),
    )
    val_groups = set(groups[: max(1, len(groups) // 5)])
    rank = lambda r: hashlib.sha256(f'{seed}:{r["id"]}'.encode()).hexdigest()
    train = sorted(
        [r for r in records["train"] if r["group"] not in val_groups], key=rank
    )[:train_count]
    val = sorted([r for r in records["train"] if r["group"] in val_groups], key=rank)[
        :val_count
    ]
    test = sorted(records["val"], key=lambda r: r["id"])
    if len(train) != train_count or len(val) != val_count or not test:
        raise ValueError(
            "Insufficient unique samples for the requested disjoint splits"
        )
    for name, rows in [("train", train), ("val", val), ("test", test)]:
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    provenance = {
        "dataset": "VIA / GGDataSet lane semantic segmentation (miniature physical driving track)",
        "url": URL,
        "archive_sha256": SHA256,
        "seed": seed,
        "available_after_exact_dedup": {k: len(v) for k, v in records.items()},
        "used": {"train": len(train), "validation": len(val), "test": len(test)},
        "validation_groups": sorted(val_groups),
        "split_policy": "100-index blocks within original train; original val reserved for test",
        "label": "RGB (192,128,128) Line versus all other pixels",
        "instance_labels": False,
        "duplicates_removed": duplicates,
        "limits": "Source sequence IDs absent; block split does not prove independent videos or circuits. Not TuSimple/CULane/BDD100K/FoggyLane.",
        "license_reference": "https://github.com/makerviet/via-dataset-cds-2020/blob/dbb2307748961cd4c5ec8743aa8e5c6ac4abc249/LICENSE (MIT; copyright Maker Hanoi 2021)",
        "redistribution": "Metrics, evaluation mask visualizations, sample hashes and original-code trained weights only; no source photographs or dataset archive redistributed.",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance["used"]))
    return provenance


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", default="data/Segmentation_Dataset.zip")
    p.add_argument("--output", default="data/via")
    p.add_argument("--download", action="store_true")
    p.add_argument("--train-count", type=int, default=256)
    p.add_argument("--val-count", type=int, default=64)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    archive = Path(a.archive)
    if a.download and not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as r, archive.open("wb") as f:
            while chunk := r.read(4 * 1024 * 1024):
                f.write(chunk)
    prepare(archive, a.output, a.train_count, a.val_count, a.seed)


if __name__ == "__main__":
    main()
