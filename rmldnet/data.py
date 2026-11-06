"""Portable JSONL dataset, augmentation and explicitly synthetic fixtures."""

import json
from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from torch.utils.data import Dataset


def fog(image, rng):
    h, w = image.shape[:2]
    depth = np.linspace(0.4, 1.8, h)[:, None, None]
    t = np.exp(-rng.uniform(0.2, 1.5) * depth)
    a = rng.uniform(0.7, 1.0)
    return (image * t + a * (1 - t)).astype(np.float32)


class LaneDataset(Dataset):
    """Paths relative to manifest; uint16 masks with 0 background, positive lane IDs."""

    def __init__(self, manifest, size=(320, 800), augment=False):
        self.path = Path(manifest)
        self.rows = [
            json.loads(l) for l in self.path.read_text().splitlines() if l.strip()
        ]
        if not self.rows:
            raise ValueError("Empty dataset")
        self.size, self.augment = size, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(self.path.parent / row["image"]).convert("RGB")
        mask = Image.open(self.path.parent / row["mask"])
        if image.size != mask.size:
            raise ValueError("Image/mask size mismatch")
        h, w = self.size
        image = image.resize((w, h), Image.Resampling.BILINEAR)
        mask = mask.resize((w, h), Image.Resampling.NEAREST)
        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.7, 1.3))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3))
            if random.random() < 0.3:
                image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 1.2)))
        clear = np.asarray(image, dtype=np.float32) / 255
        observed = (
            fog(clear, random) if self.augment and random.random() < 0.5 else clear
        )
        return {
            "image": torch.from_numpy(observed.copy()).permute(2, 0, 1),
            "clear": torch.from_numpy(clear.copy()).permute(2, 0, 1),
            "mask": torch.from_numpy(np.asarray(mask, dtype=np.int64).copy()),
            "instance_valid": bool(row.get("instance_supervision", True)),
        }


def synthetic(root, count=12, seed=7, size=(64, 128)):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    h, w = size
    for i in range(count):
        image = Image.new("RGB", (w, h), (45, 48, 50))
        mask = Image.new("I", (w, h), 0)
        draw, md = ImageDraw.Draw(image), ImageDraw.Draw(mask)
        for lane, base in enumerate((0.22, 0.5, 0.78), 1):
            bend = rng.uniform(-0.07, 0.07)
            pts = [
                (int(w * (0.5 + (base - 0.5) * (y / h) + bend * (y / h) ** 2)), y)
                for y in range(h // 4, h)
            ]
            draw.line(pts, fill=(235, 235, 210), width=3)
            md.line(pts, fill=lane, width=3)
        image.save(root / f"image_{i}.png")
        mask.convert("I;16").save(root / f"mask_{i}.png")
        rows.append(
            {
                "image": f"image_{i}.png",
                "mask": f"mask_{i}.png",
                "sequence": f"synthetic-{i}",
                "frame": 0,
            }
        )
    # Split by independent generated scene; never evaluate on the training manifest.
    split = max(1, int(count * 0.75))
    for name, subset in [("train", rows[:split]), ("val", rows[split:])]:
        if not subset:
            raise ValueError("Need at least two synthetic samples")
        (root / f"{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in subset)
        )
    (root / "PROVENANCE.txt").write_text(
        f"SYNTHETIC ONLY. seed={seed}; count={count}; no benchmark claims.\n"
    )
