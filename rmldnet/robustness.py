"""Fixed-checkpoint fog, calibration, error and corridor-geometry evaluation."""

import csv
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import LaneDataset
from .experiment import make_model, perturb, state_hash, write_results
from .geometry import extract_corridor
from .metrics import Confusion
from .public_data import digest


class CalibrationAccumulator:
    """Decision-class probability reliability for a fixed segmentation threshold."""

    def __init__(self, bins=15):
        if bins < 2:
            raise ValueError("Need at least two calibration bins")
        self.bins = bins
        self.counts = np.zeros(bins, dtype=np.int64)
        self.confidence = np.zeros(bins)
        self.correct = np.zeros(bins)
        self.brier = self.nll = 0.0
        self.pixels = self.positive = 0
        self.positive_brier = self.negative_brier = 0.0

    def update(self, probability, target, threshold):
        p = np.asarray(probability, dtype=float)
        y = np.asarray(target, dtype=bool)
        if p.shape != y.shape or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
            raise ValueError("Invalid calibration inputs")
        predicted = p > threshold
        confidence = np.where(predicted, p, 1 - p)
        correct = predicted == y
        index = np.minimum((confidence * self.bins).astype(int), self.bins - 1).ravel()
        self.counts += np.bincount(index, minlength=self.bins)
        self.confidence += np.bincount(
            index, weights=confidence.ravel(), minlength=self.bins
        )
        self.correct += np.bincount(index, weights=correct.ravel(), minlength=self.bins)
        error = (p - y) ** 2
        self.brier += float(error.sum())
        self.nll += float((-np.log(np.clip(np.where(y, p, 1 - p), 1e-7, 1))).sum())
        self.pixels += p.size
        self.positive += int(y.sum())
        self.positive_brier += float(error[y].sum())
        self.negative_brier += float(error[~y].sum())

    def compute(self):
        if not self.pixels:
            raise ValueError("No calibration observations")
        bins = []
        ece = 0.0
        for i, n in enumerate(self.counts):
            conf = self.confidence[i] / n if n else None
            accuracy = self.correct[i] / n if n else None
            if n:
                ece += n * abs(conf - accuracy) / self.pixels
            bins.append(
                {
                    "lower": i / self.bins,
                    "upper": (i + 1) / self.bins,
                    "pixels": int(n),
                    "mean_confidence": conf,
                    "accuracy": accuracy,
                }
            )
        return {
            "brier": self.brier / self.pixels,
            "negative_log_likelihood": self.nll / self.pixels,
            "ece": float(ece),
            "positive_brier": self.positive_brier / self.positive
            if self.positive
            else None,
            "negative_brier": self.negative_brier / (self.pixels - self.positive)
            if self.pixels > self.positive
            else None,
            "pixels": self.pixels,
            "positive_pixels": self.positive,
            "bins": bins,
            "definition": "Confidence is predicted-class probability under the frozen validation-selected decision threshold; entropy is predictive, not epistemic uncertainty.",
        }


def geometry_error(prediction, reference, width):
    result = {}
    for i, name in enumerate(["near", "far"]):
        p = prediction["centers"][i]
        r = reference["centers"][i]
        result[f"{name}_reference_valid"] = r is not None
        result[f"{name}_prediction_valid"] = p is not None
        result[f"{name}_error_px"] = (
            None if p is None or r is None else abs(p - r) * (width - 1)
        )
    return result


def evaluate_model(name, bundle, data, output, betas):
    source = Path(bundle) / name
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text())
    checkpoint = torch.load(source / "best.pt", map_location="cpu", weights_only=True)
    model = make_model(name)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    torch.set_num_threads(2)
    threshold = checkpoint["threshold"]
    dataset = LaneDataset(Path(data) / "test.jsonl", config["size"])
    loader = DataLoader(dataset, batch_size=4)
    ids = [row["id"] for row in dataset.rows]
    conditions = []
    calibration = []
    geometry = []
    failure_candidates = []
    example = None
    with torch.inference_mode():
        for beta in betas:
            metric = Confusion()
            per_image = []
            cal = CalibrationAccumulator()
            index = 0
            import time

            start = time.perf_counter()
            for batch in loader:
                probabilities = (
                    model(perturb(batch["image"], beta))["logits"]
                    .sigmoid()[:, 0]
                    .numpy()
                )
                targets = batch["mask"].numpy() > 0
                metric.update(probabilities > threshold, targets)
                cal.update(probabilities, targets, threshold)
                for p, target in zip(probabilities, targets):
                    m = Confusion()
                    m.update(p > threshold, target)
                    record = m.compute()
                    per_image.append(record)
                    if beta == 0:
                        predicted = extract_corridor(p, threshold)
                        reference = extract_corridor(target.astype(float), 0.5)
                        error = geometry_error(predicted, reference, p.shape[1])
                        geometry.append(
                            {
                                "id": ids[index],
                                **error,
                                "prediction": predicted,
                                "reference": reference,
                            }
                        )
                        score = record["f1"]
                        if score is not None:
                            failure_candidates.append(
                                (score, ids[index], p.copy(), target.copy())
                            )
                            failure_candidates = sorted(
                                failure_candidates, key=lambda r: (r[0], r[1])
                            )[:3]
                        if (
                            example is None
                            and predicted["both_valid"]
                            and reference["both_valid"]
                        ):
                            example = (
                                ids[index],
                                p.copy(),
                                target.copy(),
                                predicted["centers"],
                                reference["centers"],
                            )
                    index += 1
            conditions.append(
                dict(
                    metric.compute(),
                    samples=len(dataset),
                    elapsed_seconds=time.perf_counter() - start,
                    threshold=threshold,
                    beta=beta,
                    per_image=per_image,
                )
            )
            calibration.append({"beta": beta, **cal.compute()})
            print(
                json.dumps({"model": name, "beta": beta, "f1": conditions[-1]["f1"]}),
                flush=True,
            )
    old = json.loads((source / "results.json").read_text())
    original = next(r for r in old["conditions"] if r["beta"] == 0)
    replay = next(r for r in conditions if r["beta"] == 0)
    if replay["confusion"] != original["confusion"]:
        raise ValueError(f"Original confusion replay mismatch: {name}")
    config.update(
        evaluation_extension="fixed-checkpoint new corruption levels",
        timing_scope="Evaluation includes loading, calibration, and geometry; controlled network timing is separate",
        evaluation_betas=list(betas),
        checkpoint_reference=str(source / "best.pt"),
        evaluation_checkpoint_sha256=digest(source / "best.pt"),
    )
    (output / "config.json").write_text(json.dumps(config, indent=2))
    report = {
        "model": name,
        "seed": config["seed"],
        "selection": "Existing validation-selected checkpoint and threshold; no test tuning",
        "checkpoint_tensor_sha256": state_hash(model.state_dict()),
        "conditions": conditions,
    }
    write_results(output, report, ids)
    (output / "calibration.json").write_text(
        json.dumps(calibration, indent=2, allow_nan=False)
    )
    (output / "geometry.json").write_text(
        json.dumps(geometry, indent=2, allow_nan=False)
    )
    if failure_candidates:
        np.savez_compressed(
            output / "failure_cases.npz",
            probabilities=np.stack([r[2] for r in failure_candidates]),
            targets=np.stack([r[3] for r in failure_candidates]),
        )
        (output / "failure_cases.json").write_text(
            json.dumps([{"f1": r[0], "id": r[1]} for r in failure_candidates], indent=2)
        )
    if example is not None:
        np.savez_compressed(
            output / "geometry_example.npz",
            probability=example[1],
            target=example[2],
            predicted_centers=example[3],
            reference_centers=example[4],
        )
        (output / "geometry_example.json").write_text(
            json.dumps(
                {
                    "id": example[0],
                    "selection": "first held-out image with both reference and prediction anchors valid",
                },
                indent=2,
            )
        )
    return report, calibration, geometry


def run(bundle, data, output, betas=(0.0, 0.15, 0.45, 1.8, 2.1)):
    if (
        not betas
        or 0 not in betas
        or len(set(betas)) != len(betas)
        or any(not np.isfinite(b) or b < 0 for b in betas)
    ):
        raise ValueError(
            "Unique nonnegative fog levels including baseline zero required"
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in [
        "dataset_provenance.json",
        "train_manifest.jsonl",
        "val_manifest.jsonl",
        "test_manifest.jsonl",
    ]:
        shutil.copyfile(Path(bundle) / name, output / name)
    summary = []
    geometry_summary = []
    for name in ["compact", "compact_no_defog", "unet"]:
        report, calibration, geometry = evaluate_model(
            name, bundle, data, output / name, betas
        )
        for c in report["conditions"]:
            summary.append({"model": name, **c})
        for anchor in ["near", "far"]:
            ref = sum(r[f"{anchor}_reference_valid"] for r in geometry)
            pred = sum(r[f"{anchor}_prediction_valid"] for r in geometry)
            errors = [
                r[f"{anchor}_error_px"]
                for r in geometry
                if r[f"{anchor}_error_px"] is not None
            ]
            geometry_summary.append(
                {
                    "model": name,
                    "anchor": anchor,
                    "test_images": len(geometry),
                    "reference_available": ref,
                    "prediction_available": pred,
                    "reference_fraction_of_all_images": ref / len(geometry),
                    "paired_fraction_of_all_images": len(errors) / len(geometry),
                    "paired_observations": len(errors),
                    "paired_coverage_of_references": len(errors) / ref if ref else None,
                    "mean_absolute_error_px": float(np.mean(errors))
                    if errors
                    else None,
                    "median_absolute_error_px": float(np.median(errors))
                    if errors
                    else None,
                    "p95_absolute_error_px": float(np.quantile(errors, 0.95))
                    if errors
                    else None,
                }
            )
    keys = [
        "model",
        "beta",
        "samples",
        "threshold",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "lane_iou",
        "miou",
    ]
    with (output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=keys, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary)
    (output / "geometry_summary.json").write_text(
        json.dumps(geometry_summary, indent=2, allow_nan=False)
    )
    return {
        "conditions": [{k: r[k] for k in keys} for r in summary],
        "geometry": geometry_summary,
        "source_kind": "held-out miniature physical-track images with controlled synthetic corruption; no temporal claim",
    }
