"""Explicit-sequence segmentation → corridor measurements → Kalman estimates."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
import yaml
from .experiment import make_model
from .geometry import GeometryConfig, extract_corridor
from .kalman import LaneFilterConfig, LanePositionFilter


def load_sequence(path):
    path = Path(path)
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        raise ValueError("Sequence manifest must be an object")
    if document.get("source_kind") not in ("verified_video", "synthetic_controlled"):
        raise ValueError(
            "Sequence must explicitly declare verified_video or synthetic_controlled source_kind"
        )
    if not isinstance(document.get("sequence_id"), str) or not document["sequence_id"]:
        raise ValueError("A nonempty sequence_id is required")
    frames = document.get("frames", [])
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError("Sequence must have at least two frames")
    previous = None
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("Frame records must be objects")
        timestamp = frame.get("timestamp_s")
        if (
            not isinstance(timestamp, (float, int))
            or not np.isfinite(timestamp)
            or (previous is not None and timestamp <= previous)
        ):
            raise ValueError("Frame timestamps must strictly increase")
        previous = timestamp
        if not isinstance(frame.get("image"), str):
            raise ValueError("Frame image path is required")
        if not (path.parent / frame["image"]).is_file():
            raise FileNotFoundError(path.parent / frame["image"])
    return document


def run_sequence(
    manifest, experiment, output, filter_config=None, geometry_config=None
):
    manifest = Path(manifest)
    document = load_sequence(manifest)
    experiment = Path(experiment)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((experiment / "config.json").read_text())
    checkpoint = torch.load(
        experiment / "best.pt", map_location="cpu", weights_only=True
    )
    model = make_model(config["model"])
    model.load_state_dict(checkpoint["state"])
    model.eval()
    torch.set_num_threads(2)
    tracker = LanePositionFilter(filter_config)
    geometry_config = geometry_config or GeometryConfig()
    rows = []
    h, w = config["size"]
    with torch.inference_mode():
        for frame in document["frames"]:
            image = (
                Image.open(manifest.parent / frame["image"])
                .convert("RGB")
                .resize((w, h), Image.Resampling.BILINEAR)
            )
            tensor = torch.from_numpy(
                np.asarray(image, dtype=np.float32) / 255
            ).permute(2, 0, 1)[None]
            probability = model(tensor)["logits"][0, 0].sigmoid().numpy()
            measurement = extract_corridor(
                probability, checkpoint["threshold"], geometry_config
            )
            estimate = tracker.step(measurement["centers"], frame["timestamp_s"])
            rows.append(
                {
                    "timestamp_s": frame["timestamp_s"],
                    "image": frame["image"],
                    "geometry": measurement,
                    "filter": estimate,
                    "truth_centers": frame.get("truth_centers"),
                }
            )
    report = {
        "sequence_id": document["sequence_id"],
        "source_kind": document["source_kind"],
        "frames": len(rows),
        "model": config["model"],
        "threshold": checkpoint["threshold"],
        "filter_config": asdict(tracker.config),
        "geometry_config": asdict(geometry_config),
        "rows": rows,
        "limits": "Source ordering is an explicit caller assertion; no temporal ordering is inferred from dataset filenames. Coordinates are image-plane proxies.",
    }
    (output / "sequence.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    metrics = {
        "source_kind": document["source_kind"],
        "frames": len(rows),
        "estimated_frames": sum(row["filter"]["state"] is not None for row in rows),
    }
    reference_matches = document.get("truth_anchor_y") == [
        geometry_config.near_y,
        geometry_config.far_y,
    ]
    if document["source_kind"] == "synthetic_controlled" and reference_matches:
        paired = [
            row
            for row in rows
            if row["filter"]["state"] is not None
            and row["truth_centers"] is not None
            and row["geometry"]["both_valid"]
        ]
        if paired:
            truth = np.asarray([row["truth_centers"] for row in paired], dtype=float)
            raw = np.asarray(
                [row["geometry"]["centers"] for row in paired], dtype=float
            )
            filtered = np.asarray(
                [row["filter"]["state"][:2] for row in paired], dtype=float
            )
            metrics.update(
                comparable_frames=len(paired),
                raw_rmse=float(np.sqrt(np.mean((raw - truth) ** 2))),
                filtered_rmse=float(np.sqrt(np.mean((filtered - truth) ** 2))),
                units="normalized image width",
            )
    else:
        metrics["reference_not_scored_reason"] = (
            "No matching controlled-render reference anchors"
        )
    (output / "sequence_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False)
    )
    return report


def render_sequence(output, count=48, seed=7):
    """Controlled camera-like render for executable integration checks, never real video."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    frames = []
    h, w = 96, 224
    for i in range(count):
        t = i * 0.1
        near = 0.5 + 0.04 * np.sin(0.6 * t)
        far = 0.5 + 0.025 * np.sin(0.6 * t + 0.2)
        # Small known measurement jitter is added to the rendered boundaries.
        jitter = rng.normal(0, 0.004, 2)
        image = Image.new("RGB", (w, h), (50, 52, 55))
        draw = ImageDraw.Draw(image)
        ys = np.arange(int(h * 0.2), h)
        center = far + (near - far) * (ys / (h - 1) - 0.6) / (0.85 - 0.6)
        observed = (
            center
            + jitter[1]
            + (jitter[0] - jitter[1]) * (ys / (h - 1) - 0.6) / (0.85 - 0.6)
        )
        width = 0.20 + 0.65 * np.clip((ys / (h - 1) - 0.2) / 0.8, 0, 1)
        for sign in [-1, 1]:
            pts = [
                (int((c + sign * lane_width / 2) * (w - 1)), int(y))
                for c, lane_width, y in zip(observed, width, ys)
            ]
            draw.line(pts, fill=(235, 235, 225), width=3)
        name = f"frame_{i:04d}.png"
        image.save(output / name)
        frames.append(
            {
                "image": name,
                "timestamp_s": float(t),
                "truth_centers": [float(near), float(far)],
            }
        )
    path = output / "sequence.json"
    path.write_text(
        json.dumps(
            {
                "sequence_id": f"controlled-render-{seed}",
                "source_kind": "synthetic_controlled",
                "truth_anchor_y": [0.85, 0.60],
                "seed": seed,
                "frames": frames,
            },
            indent=2,
        )
    )
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest")
    parser.add_argument("--experiment", default="results/via/compact")
    parser.add_argument("--output", default="runs/temporal")
    parser.add_argument("--config", default="configs/temporal.yaml")
    parser.add_argument("--render-demo", action="store_true")
    args = parser.parse_args()
    settings = yaml.safe_load(Path(args.config).read_text())
    manifest = args.manifest
    if args.render_demo:
        manifest = render_sequence(Path(args.output) / "render")
    if not manifest:
        parser.error("--manifest or --render-demo required")
    report = run_sequence(
        manifest,
        args.experiment,
        args.output,
        LaneFilterConfig(**settings["filter"]),
        GeometryConfig(**settings["geometry"]),
    )
    print(
        json.dumps(
            {
                "frames": report["frames"],
                "source_kind": report["source_kind"],
                "estimated_frames": sum(
                    r["filter"]["state"] is not None for r in report["rows"]
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
