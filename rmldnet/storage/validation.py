"""Validate public experiment bundles before any database transaction."""

import hashlib
import json
import math
from pathlib import Path
import numpy as np


TABLES = (
    "datasets",
    "samples",
    "splits",
    "split_memberships",
    "experiments",
    "conditions",
    "observations",
    "artifacts",
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def confusion(value):
    matrix = np.asarray(value)
    if matrix.shape != (2, 2) or matrix.dtype.kind not in "iu" or (matrix < 0).any():
        raise ValueError("Confusion matrices must contain four nonnegative integers")
    if matrix.max(initial=0) > 2**53:
        raise ValueError(
            "Confusion counts exceed exact JSON/SQLite metric arithmetic range"
        )
    return tuple(int(x) for x in matrix.ravel())


def finite(value, name, lower=0, upper=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    if value < lower or (upper is not None and value > upper):
        raise ValueError(f"{name} out of range")
    return value


def _read_bundle(root):
    root = Path(root).resolve()
    provenance = json.loads((root / "dataset_provenance.json").read_text())
    manifests = {}
    used_ids = set()
    used_hashes = set()
    groups = {}
    for split in ("train", "validation", "test"):
        filename = (
            "val_manifest.jsonl" if split == "validation" else f"{split}_manifest.jsonl"
        )
        rows = [
            json.loads(line)
            for line in (root / filename).read_text().splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError(f"Empty {split} split")
        for row in rows:
            required = ("id", "image", "mask", "sha256", "group")
            if any(not isinstance(row.get(k), str) or not row[k] for k in required):
                raise ValueError("Invalid sample metadata")
            if not isinstance(row.get("instance_supervision", True), bool):
                raise ValueError("Instance flag must be Boolean")
            if row["id"] in used_ids or row["sha256"] in used_hashes:
                raise ValueError("Split overlap or duplicate content")
            if len(row["sha256"]) != 64 or any(
                c not in "0123456789abcdef" for c in row["sha256"]
            ):
                raise ValueError("Invalid sample SHA256")
            used_ids.add(row["id"])
            used_hashes.add(row["sha256"])
            if groups.get(row["group"], split) != split:
                raise ValueError("Group appears in multiple splits")
            groups[row["group"]] = split
        manifests[split] = rows
    identity = json.dumps(
        {"provenance": provenance, "manifests": manifests},
        sort_keys=True,
        separators=(",", ":"),
    )
    dataset_key = hashlib.sha256(identity.encode()).hexdigest()
    test_ids = {r["id"] for r in manifests["test"]}
    runs = []
    for directory in sorted(
        p for p in root.iterdir() if p.is_dir() and (p / "results.json").exists()
    ):
        config = json.loads((directory / "config.json").read_text())
        result = json.loads((directory / "results.json").read_text())
        if config["model"] != result["model"] or config["seed"] != result["seed"]:
            raise ValueError("Run identity mismatch")
        if (
            isinstance(result["seed"], bool)
            or not isinstance(result["seed"], int)
            or result["seed"] < 0
        ):
            raise ValueError("Seed must be a nonnegative integer")
        state = result.get("checkpoint_tensor_sha256", "")
        if len(state) != 64 or any(c not in "0123456789abcdef" for c in state):
            raise ValueError("Invalid checkpoint hash")
        rawname = result.get("per_image_metrics_file", "per_image_metrics.jsonl")
        rawpath = (directory / rawname).resolve()
        if not rawpath.is_relative_to(directory.resolve()):
            raise ValueError("Artifact path escapes experiment directory")
        observations = [
            json.loads(line)
            for line in rawpath.read_text().splitlines()
            if line.strip()
        ]
        by_beta = {}
        for row in observations:
            finite(row["beta"], "beta")
            bucket = by_beta.setdefault(row["beta"], {})
            if row["id"] not in test_ids or row["id"] in bucket:
                raise ValueError("Unknown or duplicate test observation")
            bucket[row["id"]] = confusion(row["metrics"]["confusion"])
        conditions = result["conditions"]
        seen = set()
        for condition in conditions:
            beta = finite(condition["beta"], "beta")
            finite(condition["threshold"], "threshold", 0, 1)
            finite(condition["elapsed_seconds"], "elapsed_seconds")
            if beta in seen:
                raise ValueError("Duplicate test condition")
            seen.add(beta)
            rows = by_beta.get(beta, {})
            if set(rows) != test_ids or condition["samples"] != len(test_ids):
                raise ValueError("Missing test observations")
            expected = confusion(condition["confusion"])
            if (
                tuple(np.array(list(rows.values()), dtype=np.int64).sum(0).tolist())
                != expected
            ):
                raise ValueError("Aggregate confusion mismatch")
            if sum(expected) != len(test_ids) * config["size"][0] * config["size"][1]:
                raise ValueError("Pixel count mismatch")
            tn, fp, fn, tp = expected
            checks = {
                "precision": (tp, tp + fp),
                "recall": (tp, tp + fn),
                "f1": (2 * tp, 2 * tp + fp + fn),
                "accuracy": (tp + tn, tp + tn + fp + fn),
                "lane_iou": (tp, tp + fp + fn),
            }
            ious = [
                num / den
                for num, den in [(tp, tp + fp + fn), (tn, tn + fp + fn)]
                if den
            ]
            if not ious or not math.isclose(
                condition["miou"], sum(ious) / len(ious), rel_tol=1e-10
            ):
                raise ValueError("Incorrect mIoU metric")
            for key, (num, den) in checks.items():
                actual = condition[key]
                calculated = num / den if den else None
                if (actual is None) != (calculated is None) or (
                    actual is not None
                    and not math.isclose(
                        actual, calculated, rel_tol=1e-10, abs_tol=1e-12
                    )
                ):
                    raise ValueError(f"Incorrect {key} metric")
        if set(by_beta) != seen:
            raise ValueError("Undeclared test conditions")
        run_identity = json.dumps(
            {
                "dataset": dataset_key,
                "config": config,
                "result_sha256": file_hash(directory / "results.json"),
            },
            sort_keys=True,
        )
        run_key = hashlib.sha256(run_identity.encode()).hexdigest()
        runs.append(
            {
                "directory": directory,
                "config": config,
                "result": result,
                "run_key": run_key,
                "observations": by_beta,
            }
        )
    if not runs:
        raise ValueError("No completed experiment runs found")
    return root, provenance, manifests, dataset_key, runs


def read_bundle(root):
    try:
        return _read_bundle(root)
    except (KeyError, TypeError, IndexError, json.JSONDecodeError) as error:
        raise ValueError(f"Malformed experiment bundle: {error}") from error
