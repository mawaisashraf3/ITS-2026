"""Verify released VIA records and compute paired, group-level sensitivity contrasts."""

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from pathlib import Path

import numpy as np


MODELS = ("swin_b", "unet", "compact", "compact_no_defog")
BETAS = (0.0, 0.7, 1.2)
METRICS = ("precision", "recall", "f1", "accuracy", "lane_iou", "miou")
SOURCE_COMMIT = "f99112fdd8c0f736d96a2661f3940f051b71501d"
DATABASE_SHA = "8765a5496254407f86c8d88e9232b4f2c8651d29c272bb3ab2722bc606f81e8d"
CONTRASTS = (("swin_b", "unet"), ("compact", "compact_no_defog"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def score(counts):
    tn, fp, fn, tp = counts
    return dict(zip(METRICS, (
        tp / (tp + fp), tp / (tp + fn), 2 * tp / (2 * tp + fp + fn),
        (tp + tn) / sum(counts), tp / (tp + fp + fn),
        (tp / (tp + fp + fn) + tn / (tn + fp + fn)) / 2,
    )))


def f1(counts):
    _, fp, fn, tp = np.moveaxis(counts, -1, 0)
    return 2 * tp / (2 * tp + fp + fn)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_release(root, database):
    require(hashlib.sha256(database.read_bytes()).hexdigest() == DATABASE_SHA,
            "Unexpected database hash")
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        require(conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)],
                "Database failed integrity check")
        require(not conn.execute("PRAGMA foreign_key_check").fetchall(),
                "Database failed foreign key check")
        split_rows = conn.execute("""
            SELECT p.name, s.external_id, s.group_id, s.sha256, s.instance_supervision
            FROM split_memberships m JOIN samples s ON s.id=m.sample_id
            JOIN splits p ON p.id=m.split_id
        """).fetchall()
        obs = conn.execute("""
            SELECT e.model,c.beta,s.external_id,o.tn,o.fp,o.fn,o.tp
            FROM observations o JOIN samples s ON s.id=o.sample_id
            JOIN conditions c ON c.id=o.condition_id JOIN experiments e ON e.id=c.experiment_id
        """).fetchall()
        registry = {(model, beta): (t, n, tn, fp, fn, tp) for model, beta, t, n, tn, fp, fn, tp
                    in conn.execute("""
                        SELECT e.model,c.beta,c.threshold,c.sample_count,c.tn,c.fp,c.fn,c.tp
                        FROM conditions c JOIN experiments e ON e.id=c.experiment_id
                    """)}
        run_ids = dict(conn.execute("SELECT model,state_sha256 FROM experiments"))
    raw = []
    for model in MODELS:
        path = root / "results/via" / model / "per_image_metrics.jsonl"
        for line in filter(str.strip, path.read_text(encoding="utf-8").splitlines()):
            row = json.loads(line)
            flat = tuple(x for pair in row["metrics"]["confusion"] for x in pair)
            raw.append((model, row["beta"], row["id"], *flat))
    require(sorted(raw) == sorted(obs), "Raw per-image records differ from registry")
    counts = {}
    members = {}
    for split, size, file in (("train", 256, "train"), ("validation", 64, "val"), ("test", 610, "test")):
        path = root / "results/via" / f"{file}_manifest.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        expected = {(r["id"], r["group"], r["sha256"], int(r["instance_supervision"])) for r in records}
        actual = {(i, g, h, v) for p, i, g, h, v in split_rows if p == split}
        require(len(records) == len(actual) == size and expected == actual, f"{split} mismatch")
        members[split] = {i: g for i, g, _, _ in actual}
    require(len({r[3] for r in split_rows}) == 930, "Sample hash overlap")
    require(len({(r[0], r[2]) for r in split_rows}) == len({r[2] for r in split_rows}),
            "Filename group overlap")
    require(all(r[4] == 0 for r in split_rows), "Unexpected instance labels")
    pixels = 96 * 224
    for model, beta, sample, *cm in obs:
        require(sample in members["test"] and sum(cm) == pixels, "Invalid sample record")
        counts.setdefault((model, beta), {})[sample] = cm
    require(len(obs) == 7320 and set(counts) == set(registry) == set(
        (m, b) for m in MODELS for b in BETAS), "Unexpected model-condition set")
    truth = {}
    groups = sorted(set(members["test"].values()))
    require(len(groups) == 13, "Unexpected group count")
    rows, group_counts = [], {}
    for model in MODELS:
        run_path = root / "results/via" / model
        run = load_json(run_path / "results.json")
        config = load_json(run_path / "config.json")
        require(run["model"] == config["model"] == model and run["seed"] == config["seed"] == 7,
                "Run identity mismatch")
        require(run["checkpoint_tensor_sha256"] == run_ids[model], "Checkpoint record mismatch")
        history = load_json(run_path / "history.json")
        peak = max(history, key=lambda row: row["validation_f1"])
        for expected in run["conditions"]:
            beta = expected["beta"]
            key = (model, beta)
            sample_counts = counts[key]
            require(sample_counts.keys() == members["test"].keys(), "Incomplete test coverage")
            by_group = {g: np.zeros(4, dtype=np.int64) for g in groups}
            for sample, cm in sample_counts.items():
                n_lane = cm[2] + cm[3]
                require(truth.setdefault(sample, n_lane) == n_lane,
                        f"Reference support varies for {sample}")
                by_group[members["test"][sample]] += cm
            group_counts[key] = np.array(list(by_group.values()))
            aggregate = group_counts[key].sum(axis=0).tolist()
            require(aggregate == [x for r in expected["confusion"] for x in r], "Run count mismatch")
            require(registry[key] == (expected["threshold"], expected["samples"], *aggregate),
                    "Registry condition mismatch")
            metrics = score(aggregate)
            for metric, value in metrics.items():
                require(math.isclose(value, expected[metric], abs_tol=1e-12), f"{key}: {metric}")
            require(peak["threshold"] == expected["threshold"], "Unexpected test threshold")
            rows.append(dict(model=model, beta=beta, images=len(sample_counts),
                             threshold=expected["threshold"], epoch=peak["epoch"],
                             parameters=config["parameters"], tn=aggregate[0], fp=aggregate[1],
                             fn=aggregate[2], tp=aggregate[3], **metrics))
    return rows, group_counts, groups, members


def summarize_groups(group_counts, groups, members, draws=20000, seed=8603):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(groups), size=(draws, len(groups)))
    rows = []
    for first, second in CONTRASTS:
        original_a = group_counts[first, 0.0]
        original_b = group_counts[second, 0.0]
        for beta in BETAS:
            a, b = group_counts[first, beta], group_counts[second, beta]
            for kind in ("paired_f1", "difference_in_fog_change"):
                if kind == "difference_in_fog_change" and beta == 0:
                    continue
                estimate = f1(a.sum(axis=0)) - f1(b.sum(axis=0))
                resamples = f1(a[indices].sum(axis=1)) - f1(b[indices].sum(axis=1))
                leave_out = f1(a.sum(axis=0) - a) - f1(b.sum(axis=0) - b)
                if kind == "difference_in_fog_change":
                    estimate -= f1(original_a.sum(axis=0)) - f1(original_b.sum(axis=0))
                    resamples -= (f1(original_a[indices].sum(axis=1))
                                  - f1(original_b[indices].sum(axis=1)))
                    leave_out -= (f1(original_a.sum(axis=0) - original_a)
                                  - f1(original_b.sum(axis=0) - original_b))
                lo, hi = np.percentile(resamples, [2.5, 97.5])
                rows.append(dict(contrast=kind, first=first, second=second, beta=beta,
                                 difference_pp=100*estimate, lower_pp=100*lo, upper_pp=100*hi,
                                 leave_one_out_min_pp=100*min(leave_out),
                                 leave_one_out_max_pp=100*max(leave_out),
                                 groups=len(groups), resamples=draws, seed=seed))
    by_group = [dict(model=m, beta=b, group=g, images=list(members["test"].values()).count(g),
                     tn=int(c[0]), fp=int(c[1]), fn=int(c[2]), tp=int(c[3]))
                for (m, b), cm in group_counts.items() for g, c in zip(groups, cm)]
    return rows, by_group


def verify_sensitivity(root, rows):
    with (root / "results/analysis/sensitivity.csv").open(newline="", encoding="utf-8") as stream:
        records = [dict(beta=float(r["beta"]), threshold=float(r["threshold"]),
                        tn=int(r["tn"]), fp=int(r["fp"]), fn=int(r["fn"]), tp=int(r["tp"]),
                        f1=float(r["f1"])) for r in csv.DictReader(stream)]
    fixed = []
    require(len(records) == 60 and len({(r["beta"], r["threshold"]) for r in records}) == 60,
            "Unexpected sensitivity grid")
    for r in records:
        counts = [r[k] for k in ("tn", "fp", "fn", "tp")]
        require(sum(counts) == 610 * 96 * 224 and r["fn"] + r["tp"] == rows[0]["fn"] + rows[0]["tp"],
                "Sweep reference counts differ")
        metrics = score(counts)
        require(math.isclose(metrics["f1"], r["f1"], abs_tol=1e-12), "Sweep F1 mismatch")
        if math.isclose(r["threshold"], 0.65):
            if r["beta"] in BETAS:
                original = next(x for x in rows if x["model"] == "compact" and x["beta"] == r["beta"])
                require(counts == [original[k] for k in ("tn", "fp", "fn", "tp")],
                        "Fixed-threshold sweep counts differ from original test")
            fixed.append({**r, **metrics})
    return records, fixed


def verify_intervals(root, group_counts):
    release = load_json(root / "results/analysis/statistics.json")
    require(release["replicates"] == 2000 and release["seed"] == 7, "Interval settings differ")
    size = len(next(iter(group_counts.values())))
    draws = np.random.default_rng(7).integers(0, size, size=(2000, size))
    intervals = []
    require(len(release["conditions"]) == len(group_counts), "Interval conditions differ")
    for row in release["conditions"]:
        counts = group_counts[row["model"], row["beta"]]
        value = float(f1(counts.sum(axis=0)))
        lo, hi = np.percentile(f1(counts[draws].sum(axis=1)), [2.5, 97.5])
        for actual, key in ((value, "f1"), (lo, "ci_low"), (hi, "ci_high")):
            require(math.isclose(actual, row[key], abs_tol=1e-12), f"Interval mismatch: {row}")
        intervals.append(dict(model=row["model"], beta=row["beta"], f1=value, ci_low=lo, ci_high=hi))
    return intervals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--database", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    database = args.database or args.root / "results/database/experiments.sqlite3"
    args.output.mkdir(parents=True, exist_ok=True)
    rows, group_counts, groups, members = load_release(args.root, database)
    contrasts, by_group = summarize_groups(group_counts, groups, members)
    sensitivity, fixed = verify_sensitivity(args.root, rows)
    write_csv(args.output / "metrics_verified.csv", rows)
    write_csv(args.output / "paired_contrasts.csv", contrasts)
    write_csv(args.output / "group_counts.csv", by_group)
    write_csv(args.output / "sensitivity_verified.csv", sensitivity)
    write_csv(args.output / "fixed_threshold.csv", fixed)
    write_csv(args.output / "model_intervals_verified.csv", verify_intervals(args.root, group_counts))
    result = dict(source_commit=SOURCE_COMMIT, database_sha256=DATABASE_SHA, observations=7320,
                  test_groups=len(groups), split_sizes={k:len(v) for k,v in members.items()},
                  independent_training_runs_added=0, bootstrap_seed=8603, bootstrap_draws=20000,
                  numpy=np.__version__, sensitivity_rows=60)
    (args.output / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
