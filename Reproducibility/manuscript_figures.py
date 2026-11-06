"""Render manuscript figures from verified count tables and released logs."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


NAMES = {
    "swin_b": "Swin-B + restoration",
    "unet": "U-Net",
    "compact": "Compact + restoration",
    "compact_no_defog": "Compact without restoration",
}
COLORS = dict(zip(NAMES, ("#23558A", "#7A3D81", "#107864", "#A34813")))
MARKERS = dict(zip(NAMES, ("o", "s", "^", "D")))


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save(fig, directory, stem):
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{stem}.{suffix}", dpi=400, facecolor="white",
                    bbox_inches="tight", pad_inches=0.09)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
                         "grid.color": "#DCE1E8", "grid.linewidth": .6, "pdf.fonttype": 42})

    model_stats = rows(args.analysis / "model_intervals_verified.csv")
    fig, ax = plt.subplots(figsize=(7.0, 3.0), layout="constrained")
    for offset, (model, label) in zip((-.15, -.05, .05, .15), NAMES.items()):
        r = [x for x in model_stats if x["model"] == model]
        f = [100*float(x["f1"]) for x in r]
        ax.errorbar([float(x["beta"])+offset for x in r], f,
                    yerr=[[p-100*float(x["ci_low"]) for p,x in zip(f,r)],
                          [100*float(x["ci_high"])-p for p,x in zip(f,r)]],
                    fmt=MARKERS[model], color=COLORS[model], capsize=3, label=label)
    ax.set(xlabel="Atmospheric corruption coefficient β", ylabel="Pooled pixel F1 (%)",
           xticks=[0, .7, 1.2], ylim=(72, 87))
    ax.legend(ncols=2, loc="upper center", frameon=False)
    save(fig, args.output, "Fig2_Performance")

    contrasts = rows(args.analysis / "paired_contrasts.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True, layout="constrained")
    for ax, first, title in zip(axes, ("swin_b", "compact"), ("Swin-B minus U-Net", "Restored minus unrestored compact")):
        r = [x for x in contrasts if x["first"] == first and x["contrast"] == "difference_in_fog_change"]
        y = [1, 0]
        v = [float(x["difference_pp"]) for x in r]
        lo = [p-float(x["lower_pp"]) for p,x in zip(v,r)]
        hi = [float(x["upper_pp"])-p for p,x in zip(v,r)]
        ax.axvline(0, color="#303741", linestyle="--", linewidth=1)
        ax.errorbar(v, y, xerr=[lo,hi], fmt=MARKERS[first], color=COLORS[first], capsize=3)
        ax.set(yticks=y, yticklabels=["β = 0.7", "β = 1.2"], ylim=(-.55,1.55),
               xlim=(-1.95,.25), title=title)
        ax.grid(axis="y", visible=False)
    fig.supxlabel("Change in the between-model F1 gap (percentage points)")
    save(fig, args.output, "Fig3_PairedFogChange")

    sweep = rows(args.analysis / "sensitivity_verified.csv")
    fixed = rows(args.analysis / "fixed_threshold.csv")
    fig, (a,b) = plt.subplots(1, 2, figsize=(7.2, 3.0), layout="constrained")
    for beta, color, marker in zip((0.,.6,1.2), ("#23558A","#A34813","#107864"), ("o","s","^")):
        r = [x for x in sweep if float(x["beta"]) == beta]
        a.plot([float(x["threshold"]) for x in r], [100*float(x["f1"]) for x in r],
               marker=marker, markersize=3.3, color=color, label=f"β = {beta:g}")
    a.axvline(.65, color="#343C46", linestyle="--", linewidth=1, label="Selected τ = 0.65")
    a.set(xlabel="Segmentation threshold τ", ylabel="Pooled pixel F1 (%)", title="(a) Threshold sweep",
          xticks=[.1,.3,.5,.65,.8,.9])
    a.legend(frameon=False, fontsize=8)
    for key, label, color, marker in zip(("precision","recall","f1"), ("Precision","Recall","F1"),
                                         ("#7A3D81","#A34813","#23558A"), ("s","^","o")):
        b.plot([float(x["beta"]) for x in fixed], [100*float(x[key]) for x in fixed], color=color,
               marker=marker, markersize=3.3, label=label)
    b.set(xlabel="Atmospheric corruption coefficient β", ylabel="Pooled pixel metric (%)",
          title="(b) Fixed threshold", xticks=[0,.3,.6,.9,1.2,1.5])
    b.legend(frameon=False, fontsize=8)
    save(fig, args.output, "Fig4_ThresholdAndFog")

    fig, ax = plt.subplots(figsize=(7.0, 3.0), layout="constrained")
    for model, label in NAMES.items():
        r = json.loads((args.root / "results/via" / model / "history.json").read_text(encoding="utf-8"))
        ax.plot([x["epoch"] for x in r], [100*x["validation_f1"] for x in r], color=COLORS[model],
                marker=MARKERS[model], markersize=3, label=label)
    ax.set(xlabel="Segmentation training epoch", ylabel="Validation pixel F1 (%)",
           ylim=(0,100), xlim=(.5,20.5))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.legend(ncols=2, loc="lower right", frameon=False)
    save(fig, args.output, "Fig5_ValidationHistory")


if __name__ == "__main__":
    main()
