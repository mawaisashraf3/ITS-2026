"""Use a published semantic experiment checkpoint on new camera images."""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .experiment import make_model


def predict(experiment, images, output):
    experiment = Path(experiment)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((experiment / "config.json").read_text())
    model = make_model(config["model"])
    checkpoint = torch.load(
        experiment / "best.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(checkpoint["state"])
    model.eval()
    h, w = config["size"]
    rows = []
    with torch.inference_mode():
        for i, name in enumerate(images):
            original = Image.open(name).convert("RGB")
            im = original.resize((w, h), Image.Resampling.BILINEAR)
            tensor = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255).permute(
                2, 0, 1
            )[None]
            prob = model(tensor)["logits"][0, 0].sigmoid().numpy()
            mask = prob > checkpoint["threshold"]
            # Return the mask at original dimensions; probability remains at network resolution.
            binary = Image.fromarray((mask * 255).astype("uint8")).resize(
                original.size, Image.Resampling.NEAREST
            )
            binary.save(output / f"{i:06d}_mask.png")
            Image.fromarray((prob * 255).astype("uint8")).save(
                output / f"{i:06d}_prob.png"
            )
            arr = np.asarray(original).copy()
            selection = np.asarray(binary) > 0
            arr[selection] = (
                arr[selection].astype(float) * 0.5 + np.array([0, 255, 80]) * 0.5
            ).astype("uint8")
            Image.fromarray(arr).save(output / f"{i:06d}_overlay.png")
            rows.append(
                {
                    "image": str(name),
                    "mask": f"{i:06d}_mask.png",
                    "threshold": checkpoint["threshold"],
                    "model": config["model"],
                    "task": "semantic lane pixels; no lane instance or tracking claim",
                }
            )
    (output / "predictions.json").write_text(json.dumps(rows, indent=2))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--output", default="runs/predictions")
    a = p.parse_args()
    torch.set_num_threads(2)
    predict(a.experiment, a.images, a.output)


if __name__ == "__main__":
    main()
