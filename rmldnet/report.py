"""Aggregate completed experiment outputs without copying literature scores."""

import argparse
import csv
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--data", default="data/via")
    p.add_argument("--output", default="results/via")
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in map(Path, a.runs):
        result = json.loads((run / "results.json").read_text())
        dest = out / run.name
        dest.mkdir(exist_ok=True)
        for name in [
            "results.json",
            "per_image_metrics.jsonl",
            "config.json",
            "history.json",
            "restoration_history.json",
            "metrics.csv",
            "learning.svg",
            "masks.png",
        ]:
            if (run / name).exists():
                shutil.copyfile(run / name, dest / name)
        for metric in result["conditions"]:
            rows.append(
                dict(
                    model=result["model"],
                    seed=result["seed"],
                    **{
                        k: v
                        for k, v in metric.items()
                        if k not in ["per_image", "confusion", "protocol"]
                    },
                )
            )
        # Only original, small-model checkpoints fit regular Git. Swin-B weights are regenerated.
        if result["model"] != "swin_b":
            shutil.copyfile(run / "best.pt", dest / "best.pt")
    keys = [
        "model",
        "seed",
        "beta",
        "samples",
        "threshold",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "lane_iou",
        "miou",
        "elapsed_seconds",
    ]
    with (out / "comparison.csv").open("w") as f:
        w = csv.DictWriter(
            f, fieldnames=keys, extrasaction="ignore", lineterminator="\n"
        )
        w.writeheader()
        w.writerows(rows)
    table = [
        "# Measured VIA lane-segmentation results",
        "",
        "Pixel metrics, not official lane-instance scores. Original held-out validation images are used as the test set; the training subset and model-selection validation subset come from separate filename blocks of original training data.",
        "",
        "| Model | Seed | Synthetic fog beta | Test images | Precision | Recall | F1 | Lane IoU | mIoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:

        def fmt(k):
            return f"{100*r[k]:.2f}" if r[k] is not None else "undefined"

        table.append(
            f'| {r["model"]} | {r["seed"]} | {r["beta"]} | {r["samples"]} | {fmt("precision")} | {fmt("recall")} | {fmt("f1")} | {fmt("lane_iou")} | {fmt("miou")} |'
        )
    table += [
        "",
        "Metrics in percent. Beta 0 uses original images; beta 0.7/1.2 are fixed atmospheric synthetic corruptions applied to the same held-out images. These are not real-fog benchmarks.",
        "",
        "The compact defogging ablation does not show a consistent benefit: its synthetic-fog F1 scores are slightly lower with defogging enabled. This staged experiment does not establish a general defogging benefit.",
        "",
        "The dataset contains miniature physical-track scenes, not full-size public-road driving. Instance identities are absent: instance loss is disabled and temporal tracking is not scored. TuSimple, CULane, BDD100K and FoggyLane are separate evaluation targets.",
        "",
        "All thresholds and best epochs are selected only on validation. Models share the same 96×224 evaluation resolution and split; budgets and initialization differ and are recorded in each config.json. Full Swin-B uses ImageNet-1K pretrained initialization; compact/U-Net start randomly. No superiority or statistical-significance claim follows from a single seed.",
        "",
        "Each model directory includes config.json, history.json, metrics.csv, learning.svg and masks.png. Mask columns are ground truth, prediction and error map (green TP, red FN, blue FP); source photographs are excluded.",
        "",
        "Small-model checkpoints are provided. The full Swin-B checkpoint exceeds regular Git size limits and is retained in the working environment; regenerate with the documented command. Source photographs and the dataset archive are not included.",
    ]
    (out / "README.md").write_text("\n".join(table) + "\n")
    shutil.copyfile(Path(a.data) / "provenance.json", out / "dataset_provenance.json")
    for split in ["train", "val", "test"]:
        shutil.copyfile(
            Path(a.data) / f"{split}.jsonl", out / f"{split}_manifest.jsonl"
        )
    # Standalone vector comparison chart with a fixed, interpretable scale.
    colors = ["#1261a0", "#c75a19", "#447c38", "#8255a2"]
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="520"><rect width="900" height="520" fill="white"/><text x="40" y="28" font-size="20">Measured held-out pixel F1 — VIA miniature-track data</text>'
    ]
    for i, model in enumerate(dict.fromkeys(r["model"] for r in rows)):
        group = [r for r in rows if r["model"] == model]
        for j, r in enumerate(group):
            x = 90 + j * 240 + i * 43
            h = 350 * (r["f1"] or 0)
            svg.append(
                f'<rect x="{x}" y="{420-h:.1f}" width="36" height="{h:.1f}" fill="{colors[i%4]}"/><text x="{x}" y="{410-h:.1f}" font-size="10">{100*(r["f1"] or 0):.1f}</text>'
            )
        svg.append(
            f'<text x="{40+i*215}" y="495" fill="{colors[i%4]}" font-size="14">{model}</text>'
        )
    for j, beta in enumerate([0, 0.7, 1.2]):
        svg.append(
            f'<text x="{100+j*240}" y="455" font-size="16">Fog beta {beta}</text>'
        )
    svg.append(
        '<path d="M60 70 V420 H850" fill="none" stroke="black"/><text x="20" y="80">1.0</text><text x="20" y="425">0.0</text></svg>'
    )
    (out / "comparison.svg").write_text("".join(svg))
    print("\n".join(table))


if __name__ == "__main__":
    main()
