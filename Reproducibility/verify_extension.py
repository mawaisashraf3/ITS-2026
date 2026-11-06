"""Reconcile the v0.4.0 held-out diagnostics and numerical filter verification."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from manuscript_analysis import load_json, require, score, write_csv


EXTENSION_COMMIT = "b5094f2dc7ba417ab1a3ecdfc3bf65455ac36dff"
MODELS = ("compact", "compact_no_defog", "unet")
BETAS = (0.0, 0.15, 0.45, 1.8, 2.1)


def jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def verify_fog(root):
    ext = root / "results/extensions/robustness"
    via = root / "results/via"
    expected = jsonl(via / "test_manifest.jsonl")
    require(expected == jsonl(ext / "test_manifest.jsonl"), "Extension split differs")
    ids = {row["id"] for row in expected}
    require(len(ids) == 610, "Unexpected test size")
    rows = []
    for model in MODELS:
        report = load_json(ext / model / "results.json")
        original = load_json(via / model / "results.json")
        require(report["checkpoint_tensor_sha256"] == original["checkpoint_tensor_sha256"],
                "Checkpoint identity differs")
        require(tuple(c["beta"] for c in report["conditions"]) == BETAS,
                "Unexpected extension condition grid")
        extension = {beta: {} for beta in BETAS}
        for row in jsonl(ext / model / "per_image_metrics.jsonl"):
            target = extension[row["beta"]]
            require(row["id"] not in target, "Duplicate observation")
            target[row["id"]] = [x for pair in row["metrics"]["confusion"] for x in pair]
        original_counts = {
            row["id"]: [x for pair in row["metrics"]["confusion"] for x in pair]
            for row in jsonl(via / model / "per_image_metrics.jsonl") if row["beta"] == 0
        }
        require(original_counts == extension[0.0], "Original-image replay differs")
        for summary in report["conditions"]:
            beta = summary["beta"]
            by_id = extension[beta]
            require(by_id.keys() == ids, "Coverage differs")
            for sample, counts in by_id.items():
                require(sum(counts) == 96 * 224 and sum(counts[2:]) == sum(original_counts[sample][2:]),
                        "Reference pixels differ")
            counts = np.array(list(by_id.values()), dtype=np.int64).sum(axis=0).tolist()
            require(counts == [x for pair in summary["confusion"] for x in pair], "Aggregate differs")
            require(summary["threshold"] == original["conditions"][0]["threshold"], "Threshold differs")
            metrics = score(counts)
            require(all(math.isclose(value, summary[key], abs_tol=1e-12)
                        for key, value in metrics.items()), "Released metric differs")
            rows.append(dict(model=model, beta=beta, images=len(by_id),
                             threshold=summary["threshold"], tn=counts[0], fp=counts[1],
                             fn=counts[2], tp=counts[3], **metrics))
    return rows


def verify_latency(root):
    records = load_json(root / "results/extensions/efficiency/efficiency.json")["models"]
    require(len({row["input_tensor_sha256"] for row in records}) == 1, "Timing input differs")
    rows = []
    for r in records:
        raw = [point["milliseconds"] for point in r["timings"]]
        require(len(raw) == 90 and r["threads"] == 2 and r["batch_size"] == 1
                and r["size"] == [96, 224], "Timing protocol differs")
        median, p95 = np.percentile(raw, [50, 95])
        require(math.isclose(median, r["latency_median_ms"], abs_tol=1e-10)
                and math.isclose(p95, r["latency_p95_ms"], abs_tol=1e-10), "Timing summary differs")
        rows.append(dict(model=r["model"], parameters=r["parameters"], median_ms=median,
                         p95_ms=p95, sampled_rss_mib=r["sampled_peak_inference_rss_mb"]))
    return rows


def summarize_kalman(path):
    with path.open(newline="", encoding="utf-8") as stream:
        runs = list(csv.DictReader(stream))
    rows = []
    for stress in ("False", "True"):
        group = [row for row in runs if row["stress"] == stress]
        require(sorted(int(row["seed"]) for row in group) == list(range(207, 227)),
                "Unexpected verification seeds")
        fields = ("raw_rmse_observed", "filtered_rmse_observed", "filtered_rmse_all",
                  "ema_rmse_observed", "position_95_interval_coverage")
        summary = {field: float(np.mean([float(row[field]) for row in group])) for field in fields}
        rows.append(dict(condition="stress" if stress == "True" else "nominal",
                         seeds=len(group), **summary))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = verify_fog(args.root)
    latency = verify_latency(args.root)
    kalman = summarize_kalman(
        args.root / "results/extensions/kalman/kalman_verification_metrics.csv"
    )
    write_csv(args.output / "extended_fog_verified.csv", rows)
    write_csv(args.output / "latency_verified.csv", latency)
    write_csv(args.output / "kalman_summary_verified.csv", kalman)
    print(f"{EXTENSION_COMMIT}: verified {len(rows)} conditions, {610*len(rows)} per-image "
          f"records, {90*len(latency)} forward times and {sum(r['seeds'] for r in kalman)} "
          "synthetic nominal/stress verification runs.")


if __name__ == "__main__":
    main()
