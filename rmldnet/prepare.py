"""Convert licensed local annotations to a common instance-mask manifest."""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def bdd_vertices(poly):
    vertices = poly["vertices"]
    types = poly.get("types", "L" * len(vertices))
    result = [vertices[0]]
    i = 1
    while i < len(vertices):
        if types[i] == "C":
            if i + 2 >= len(vertices):
                raise ValueError("Incomplete BDD cubic segment")
            p = np.asarray([result[-1], *vertices[i : i + 3]], float)
            for t in np.linspace(0, 1, 20)[1:]:
                result.append(
                    (
                        (1 - t) ** 3 * p[0]
                        + 3 * (1 - t) ** 2 * t * p[1]
                        + 3 * (1 - t) * t * t * p[2]
                        + t**3 * p[3]
                    ).tolist()
                )
            i += 3
        else:
            result.append(vertices[i])
            i += 1
    return result


def convert(kind, annotations, images, output, width=5):
    annotations, images, output = (
        Path(annotations),
        Path(images).resolve(),
        Path(output).resolve(),
    )
    output.mkdir(parents=True, exist_ok=True)
    records = []
    if kind == "tusimple":
        for line in annotations.read_text().splitlines():
            row = json.loads(line)
            lanes = []
            for xs in row["lanes"]:
                lanes.append([(x, y) for x, y in zip(xs, row["h_samples"]) if x >= 0])
            records.append((row["raw_file"], lanes))
    elif kind == "culane":
        for line in annotations.read_text().splitlines():
            name = line.split()[0].lstrip("/")
            label = images / Path(name).with_suffix(".lines.txt")
            lanes = []
            for coords in label.read_text().splitlines():
                a = [float(x) for x in coords.split()]
                if len(a) % 2:
                    raise ValueError(f"Odd coordinates in {label}")
                lanes.append(list(zip(a[::2], a[1::2])))
            records.append((name, lanes))
    elif kind == "bdd100k":
        for row in json.loads(annotations.read_text()):
            lanes = []
            for label in row.get("labels", []):
                if label.get("category", "").startswith("lane"):
                    for poly in label.get("poly2d", []):
                        lanes.append(bdd_vertices(poly))
            records.append((row["name"], lanes))
    else:
        raise ValueError(
            "Use canonical manifests for FoggyLane; no verified native schema supplied"
        )
    manifest = []
    for i, (name, lanes) in enumerate(records):
        path = images / name
        with Image.open(path) as im:
            size = im.size
        mask = Image.new("I", size, 0)
        draw = ImageDraw.Draw(mask)
        for lane, pts in enumerate(lanes, 1):
            if len(pts) > 1:
                draw.line([tuple(p) for p in pts], fill=lane, width=width)
        mask.convert("I;16").save(output / f"{i:07d}.png")
        manifest.append({"image": str(path), "mask": f"{i:07d}.png"})
    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in manifest)
    )
    (output / "conversion.json").write_text(
        json.dumps(
            {
                "format": kind,
                "line_width_original_pixels": width,
                "samples": len(manifest),
            },
            indent=2,
        )
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("format", choices=["tusimple", "culane", "bdd100k"])
    p.add_argument("--annotations", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--width", type=int, default=5)
    a = p.parse_args()
    convert(a.format, a.annotations, a.images, a.output, a.width)


if __name__ == "__main__":
    main()
