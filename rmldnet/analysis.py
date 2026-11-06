"""Cluster-bootstrap uncertainty and threshold/fog sensitivity from actual data."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import LaneDataset
from .experiment import make_model, perturb
from .storage import ExperimentStore


def f1_counts(counts):
    counts = np.asarray(counts, dtype=float)
    den = 2 * counts[..., 3] + counts[..., 1] + counts[..., 2]
    return np.divide(
        2 * counts[..., 3], den, out=np.full_like(den, np.nan), where=den > 0
    )


def cluster_bootstrap(group_counts, replicates=2000, seed=7, confidence=0.95):
    """Resample entire source-index groups; preserve paired models in trailing axes."""
    values = np.asarray(group_counts, dtype=float)
    if values.ndim not in (2, 3) or values.shape[-1] != 4 or len(values) < 2:
        raise ValueError("Need at least two groups of confusion counts")
    if replicates < 100 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap configuration")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), (replicates, len(values)))
    scores = f1_counts(values[draws].sum(axis=1))
    alpha = (1 - confidence) / 2
    return {
        "point": f1_counts(values.sum(axis=0)).tolist(),
        "low": np.nanquantile(scores, alpha, axis=0).tolist(),
        "high": np.nanquantile(scores, 1 - alpha, axis=0).tolist(),
    }, scores


def statistics(database, output, replicates=2000, seed=7):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    pair = None
    with ExperimentStore(database) as store:
        conditions = store.metrics()
        for condition in conditions:
            groups = store.query(
                """SELECT s.group_id,sum(o.tn) tn,sum(o.fp) fp,sum(o.fn) fn,sum(o.tp) tp
              FROM observations o JOIN samples s ON s.id=o.sample_id WHERE o.condition_id=? GROUP BY s.group_id ORDER BY s.group_id""",
                (condition["condition_id"],),
            )
            counts = np.array(
                [[g[k] for k in ["tn", "fp", "fn", "tp"]] for g in groups]
            )
            result, _ = cluster_bootstrap(counts, replicates, seed)
            rows.append(
                {
                    "model": condition["model"],
                    "seed": condition["seed"],
                    "beta": condition["beta"],
                    "groups": len(groups),
                    "f1": result["point"],
                    "ci_low": result["low"],
                    "ci_high": result["high"],
                }
            )
        baseline = [c for c in conditions if c["model"] == "unet" and c["beta"] == 0]
        main = [c for c in conditions if c["model"] == "swin_b" and c["beta"] == 0]
        if len(main) == len(baseline) == 1:
            pairs = []
            for condition in [main[0], baseline[0]]:
                groups = store.query(
                    """SELECT s.group_id,sum(o.tn) tn,sum(o.fp) fp,sum(o.fn) fn,sum(o.tp) tp
                FROM observations o JOIN samples s ON s.id=o.sample_id WHERE o.condition_id=? GROUP BY s.group_id ORDER BY s.group_id""",
                    (condition["condition_id"],),
                )
                pairs.append(groups)
            if [g["group_id"] for g in pairs[0]] != [g["group_id"] for g in pairs[1]]:
                raise ValueError("Paired groups differ")
            values = np.array(
                [
                    [[g[k] for k in ["tn", "fp", "fn", "tp"]] for g in group]
                    for group in pairs
                ]
            ).transpose(1, 0, 2)
            result, scores = cluster_bootstrap(values, replicates, seed)
            differences = scores[:, 0] - scores[:, 1]
            pair = {
                "comparison": "swin_b minus unet, original images",
                "delta_f1": result["point"][0] - result["point"][1],
                "ci_low": float(np.nanquantile(differences, 0.025)),
                "ci_high": float(np.nanquantile(differences, 0.975)),
            }
    report = {
        "method": "percentile cluster bootstrap of 100-index source filename groups",
        "replicates": replicates,
        "seed": seed,
        "confidence": 0.95,
        "conditions": rows,
        "paired_comparison": pair,
        "limits": "Groups are filename blocks, not verified independent drives. One training seed; intervals do not establish road-domain generalization or repeated-training uncertainty.",
    }
    (output / "statistics.json").write_text(json.dumps(report, indent=2))
    with (output / "statistics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return report


def sensitivity(experiment, data, output):
    """Descriptive held-out sweeps; do not change the selected checkpoint/threshold."""
    root = Path(experiment)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((root / "config.json").read_text())
    checkpoint = torch.load(root / "best.pt", map_location="cpu", weights_only=True)
    model = make_model(config["model"])
    model.load_state_dict(checkpoint["state"])
    model.eval()
    torch.set_num_threads(2)
    dataset = LaneDataset(Path(data) / "test.jsonl", config["size"])
    loader = DataLoader(dataset, batch_size=4)
    thresholds = np.array(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, checkpoint["threshold"], 0.7, 0.8, 0.9]
    )
    thresholds = np.unique(thresholds)
    rows = []
    with torch.inference_mode():
        for beta in [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]:
            matrices = np.zeros((len(thresholds), 4), dtype=np.int64)
            for batch in loader:
                probabilities = (
                    model(perturb(batch["image"], beta))["logits"]
                    .sigmoid()[:, 0]
                    .numpy()
                )
                target = batch["mask"].numpy() > 0
                for i, threshold in enumerate(thresholds):
                    pred = probabilities > threshold
                    matrices[i] += [
                        np.sum(~pred & ~target),
                        np.sum(pred & ~target),
                        np.sum(~pred & target),
                        np.sum(pred & target),
                    ]
            for threshold, counts, score in zip(
                thresholds, matrices, f1_counts(matrices)
            ):
                rows.append(
                    {
                        "beta": beta,
                        "threshold": float(threshold),
                        "f1": float(score),
                        "tn": int(counts[0]),
                        "fp": int(counts[1]),
                        "fn": int(counts[2]),
                        "tp": int(counts[3]),
                    }
                )
    report = {
        "model": config["model"],
        "samples": len(dataset),
        "fixed_validation_threshold": checkpoint["threshold"],
        "purpose": "descriptive held-out sensitivity; no selection or retraining",
        "rows": rows,
    }
    (output / "sensitivity.json").write_text(json.dumps(report, indent=2))
    with (output / "sensitivity.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return report


def figures(bundle, analysis_dir, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stats = json.loads((Path(analysis_dir) / "statistics.json").read_text())
    conditions = stats["conditions"]
    names = list(dict.fromkeys(r["model"] for r in conditions))
    colors = ["#2563a6", "#e28427", "#33936f", "#8b5aa6"]
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    for i, name in enumerate(names):
        selected = [r for r in conditions if r["model"] == name]
        x = np.arange(len(selected)) + i * 0.19 - 0.285
        y = np.array([r["f1"] for r in selected])
        low = np.array([r["ci_low"] for r in selected])
        high = np.array([r["ci_high"] for r in selected])
        ax.bar(x, y * 100, width=0.18, label=name, color=colors[i])
        ax.errorbar(
            x,
            y * 100,
            yerr=np.stack([y - low, high - y]) * 100,
            fmt="none",
            ecolor="#333333",
            capsize=2,
        )
    ax.set(
        xticks=[0, 1, 2],
        xticklabels=["Original", "Fog β=0.7", "Fog β=1.2"],
        ylabel="Pixel F1 (%)",
        ylim=(0, 100),
        title="Held-out lane segmentation: 610 miniature-track images",
    )
    ax.legend(ncol=2, loc="lower left")
    ax.grid(axis="y", alpha=0.15)
    save_figure(fig, output / "performance")
    plt.close(fig)
    sweep = json.loads((Path(analysis_dir) / "sensitivity.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for beta in [0, 0.6, 1.2]:
        selected = [r for r in sweep["rows"] if r["beta"] == beta]
        axes[0].plot(
            [r["threshold"] for r in selected],
            [100 * r["f1"] for r in selected],
            marker="o",
            label=f"β={beta}",
        )
    selected = [
        r
        for r in sweep["rows"]
        if r["threshold"] == sweep["fixed_validation_threshold"]
    ]
    axes[1].plot(
        [r["beta"] for r in selected],
        [100 * r["f1"] for r in selected],
        marker="o",
        color="#2563a6",
    )
    axes[0].axvline(
        sweep["fixed_validation_threshold"],
        linestyle="--",
        color="#666666",
        label="Validation-selected threshold",
    )
    axes[0].set(
        xlabel="Segmentation threshold",
        ylabel="Pixel F1 (%)",
        title="Threshold sensitivity",
    )
    axes[0].legend(fontsize=8)
    axes[1].set(
        xlabel="Atmospheric corruption β",
        ylabel="Pixel F1 (%)",
        title="Fixed-threshold fog sensitivity",
    )
    for ax in axes:
        ax.grid(alpha=0.15)
    save_figure(fig, output / "sensitivity")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for i, name in enumerate(names):
        history = json.loads((Path(bundle) / name / "history.json").read_text())
        ax.plot(
            [r["epoch"] for r in history],
            [100 * r["validation_f1"] for r in history],
            marker=".",
            label=name,
            color=colors[i],
        )
    ax.set(
        xlabel="Training epoch",
        ylabel="Validation pixel F1 (%)",
        ylim=(0, 100),
        title="Observed model-selection histories; training budgets differ",
    )
    ax.legend()
    ax.grid(alpha=0.15)
    save_figure(fig, output / "training")
    plt.close(fig)


def save_figure(fig, stem, dpi=220):
    for extension in ["png", "svg", "pdf"]:
        fig.savefig(f"{stem}.{extension}", dpi=dpi, metadata={"Creator": "RMLDNet"})
        if extension == "svg":
            path = Path(f"{stem}.svg")
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["statistics", "sensitivity", "figures"])
    parser.add_argument("--database", default="runs/platform.sqlite3")
    parser.add_argument("--output", default="results/analysis")
    parser.add_argument("--experiment", default="results/via/compact")
    parser.add_argument("--data", default="data/via")
    parser.add_argument("--bundle", default="results/via")
    parser.add_argument("--analysis", default="results/analysis")
    args = parser.parse_args()
    if args.command == "statistics":
        statistics(args.database, args.output)
    elif args.command == "sensitivity":
        sensitivity(args.experiment, args.data, args.output)
    else:
        figures(args.bundle, args.analysis, args.output)


if __name__ == "__main__":
    main()
