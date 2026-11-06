"""Reproducible figures and captions for six additional experiment groups."""

import json
from pathlib import Path
import numpy as np
from .analysis import save_figure


def load(root, path):
    return json.loads((Path(root) / path).read_text())


def create_figures(root):
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
    root = Path(root)
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    colors = ["#2563a6", "#db8126", "#31906c", "#9257a3"]

    def finish(fig, name):
        save_figure(fig, out / name, dpi=300)
        plt.close(fig)

    nominal = load(root, "kalman/nominal_trajectory.json")
    stress = load(root, "kalman/stress_trajectory.json")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), layout="constrained")
    for i, ax in enumerate(axes[:2]):
        t = np.array([r["time_s"] for r in nominal])
        truth = np.array([r["truth"][i] for r in nominal])
        raw = np.array([r["measurement"][i] for r in nominal])
        pred = np.array(
            [
                np.nan if r["predicted_state"] is None else r["predicted_state"][i]
                for r in nominal
            ]
        )
        filtered = np.array(
            [np.nan if r["state"] is None else r["state"][i] for r in nominal]
        )
        std = np.sqrt(
            [
                np.nan if r["covariance"] is None else r["covariance"][i][i]
                for r in nominal
            ]
        )
        ax.scatter(t, raw, s=8, color="#999999", alpha=0.65, label="Noisy measurement")
        ax.plot(t, truth, color="black", label="Known synthetic truth")
        ax.plot(t, pred, "--", color=colors[1], lw=1, label="Prior prediction")
        ax.plot(t, filtered, color=colors[0], label="Posterior update")
        ax.fill_between(
            t,
            filtered - 1.96 * std,
            filtered + 1.96 * std,
            color=colors[0],
            alpha=0.18,
            label="Posterior ±1.96σ",
        )
        ax.set(
            ylabel=f'{["Near","Far"][i]} center / width',
            title=f'SYNTHETIC trajectory — {["near","far"][i]} anchor',
        )
        ax.grid(alpha=0.15)
    sequence = load(root, "temporal/sequence.json")["rows"]
    st = np.array([r["timestamp_s"] for r in sequence])
    sm = np.array(
        [
            np.nan
            if r["geometry"]["centers"][0] is None
            else r["geometry"]["centers"][0]
            for r in sequence
        ]
    )
    sf = np.array(
        [
            np.nan if r["filter"]["state"] is None else r["filter"]["state"][0]
            for r in sequence
        ]
    )
    truth = np.array([r["truth_centers"][0] for r in sequence])
    axes[2].plot(st, truth, color="black", label="Controlled-render truth")
    axes[2].scatter(
        st, sm, s=12, color="#999999", label="Trained-model mask measurement"
    )
    axes[2].plot(st, sf, color=colors[0], label="Filtered image-derived measurement")
    axes[2].set(
        ylabel="Near center / width",
        title="SYNTHETIC image sequence → trained segmentation → filter",
    )
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.15)
    axes[0].legend(ncol=3, fontsize=8)
    for ax in axes:
        ax.set_xlabel("Time (s)")
    finish(fig, "kalman_nominal")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    ax = axes[0]
    t = np.array([r["time_s"] for r in stress])
    raw = np.array(
        [np.nan if r["measurement"][0] is None else r["measurement"][0] for r in stress]
    )
    truth = np.array([r["truth"][0] for r in stress])
    filtered = np.array(
        [np.nan if r["state"] is None else r["state"][0] for r in stress]
    )
    std = np.sqrt(
        [np.nan if r["covariance"] is None else r["covariance"][0][0] for r in stress]
    )
    missing = np.array([r["missing"] for r in stress])
    outlier = np.array([r["outlier"] for r in stress])
    ax.scatter(t, raw, s=6, color="#999999", label="Observed")
    ax.scatter(
        t[outlier], raw[outlier], s=12, color="#c3443e", label="Injected outlier"
    )
    ax.plot(t, truth, color="black", label="Known truth")
    ax.plot(t, filtered, color=colors[0], label="Gated filter")
    ax.fill_between(
        t, filtered - 1.96 * std, filtered + 1.96 * std, color=colors[0], alpha=0.2
    )
    ax.fill_between(
        t,
        0,
        1,
        where=missing,
        transform=ax.get_xaxis_transform(),
        color="#555555",
        alpha=0.10,
        label="Missing observation",
    )
    ax.set(
        xlabel="Time (s)",
        ylabel="Near center / width",
        title="SYNTHETIC dropout/outlier stress",
    )
    ax.legend(fontsize=6)
    report = load(root, "kalman/kalman_results.json")
    rows = [r for r in report["verification_scenarios"] if r["stress"]]
    fields = ["raw_rmse_observed", "ema_rmse_observed", "filtered_rmse_observed"]
    values = [[r[k] for r in rows] for k in fields]
    axes[1].bar(
        ["Raw", "EMA", "Kalman"],
        [np.mean(v) for v in values],
        yerr=[np.std(v, ddof=1) for v in values],
        color=colors[:3],
        capsize=4,
    )
    axes[1].set(
        ylabel="Observed-frame RMSE / width",
        title=f"{len(rows)} verification noise seeds (mean ± SD)",
    )
    sweep = report["process_noise_sensitivity"]
    axes[2].semilogx(
        [r["filter_config"]["acceleration_variance"] for r in sweep],
        [r["filtered_rmse_all"] for r in sweep],
        marker="o",
        color=colors[0],
    )
    axes[2].set(
        xlabel="Acceleration variance q",
        ylabel="All-frame RMSE / width",
        title="SYNTHETIC q sensitivity (seed 7)",
    )
    for ax in axes:
        ax.grid(alpha=0.15)
    finish(fig, "kalman_stress")
    names = ["compact", "compact_no_defog", "unet"]
    short = ["Compact + restoration", "Compact, no restoration", "U-Net"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    fog = []
    for name, label, color in zip(names, short, colors):
        conditions = load(root, f"robustness/{name}/results.json")["conditions"]
        fog.append(conditions)
        axes[0].plot(
            [r["beta"] for r in conditions],
            [100 * r["f1"] for r in conditions],
            marker="o",
            label=label,
            color=color,
        )
    delta = [100 * (a["f1"] - b["f1"]) for a, b in zip(fog[0], fog[1])]
    axes[1].plot([r["beta"] for r in fog[0]], delta, marker="o", color=colors[0])
    axes[1].axhline(0, color="black", lw=0.8)
    axes[0].set(
        xlabel="Synthetic atmospheric β",
        ylabel="Held-out pixel F1 (%)",
        title="Frozen-checkpoint extended fog comparison",
    )
    axes[0].legend(fontsize=8)
    axes[1].set(
        xlabel="Synthetic atmospheric β",
        ylabel="F1 difference (percentage points)",
        title="Restoration-enabled minus disabled",
    )
    for ax in axes:
        ax.grid(alpha=0.15)
    finish(fig, "fog_robustness")
    efficiency = load(root, "efficiency/efficiency.json")["models"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    labels = [r["model"] for r in efficiency]
    x = np.arange(len(labels))
    median = np.array([r["latency_median_ms"] for r in efficiency])
    low = np.array(
        [
            np.quantile([t["milliseconds"] for t in r["timings"]], 0.25)
            for r in efficiency
        ]
    )
    high = np.array([r["latency_p95_ms"] for r in efficiency])
    axes[0].bar(
        x, median, yerr=np.stack([median - low, high - median]), capsize=3, color=colors
    )
    axes[0].set(
        ylabel="Forward latency (ms, log scale)",
        title="Median with P25–P95 timing range",
        yscale="log",
    )
    axes[1].bar(x, [r["parameters"] / 1e6 for r in efficiency], color=colors)
    axes[1].set(
        ylabel="Parameters (millions, log scale)",
        yscale="log",
        title="Actual model size",
    )
    axes[2].bar(
        x, [r["sampled_peak_inference_rss_mb"] for r in efficiency], color=colors
    )
    axes[2].set(
        ylabel="Sampled total process RSS (MiB)",
        title="Process memory; not activation-only",
    )
    for ax in axes:
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.15)
    finish(fig, "efficiency")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    calibrations = []
    for name, label, color in zip(names, short, colors):
        cal = next(
            r
            for r in load(root, f"robustness/{name}/calibration.json")
            if r["beta"] == 0
        )
        calibrations.append(cal)
        bins = [b for b in cal["bins"] if b["pixels"]]
        axes[0].plot(
            [b["mean_confidence"] for b in bins],
            [b["accuracy"] for b in bins],
            marker="o",
            color=color,
            label=f'{label}: ECE={cal["ece"]:.3f}',
        )
    axes[0].plot([0, 1], [0, 1], "--", color="#777777")
    axes[0].set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean selected-class probability",
        ylabel="Observed correctness",
        title="Reliability; background-heavy pixel population",
    )
    axes[0].legend(fontsize=7)
    x = np.arange(3)
    axes[1].bar(
        x - 0.18,
        [r["positive_brier"] for r in calibrations],
        width=0.36,
        color=colors[1],
        label="Lane pixels",
    )
    axes[1].bar(
        x + 0.18,
        [r["negative_brier"] for r in calibrations],
        width=0.36,
        color=colors[0],
        label="Background pixels",
    )
    axes[1].set_xticks(x, short, rotation=15, ha="right")
    axes[1].set(
        ylabel="Mean squared probability error", title="Class-conditional Brier error"
    )
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.15)
    finish(fig, "confidence")
    geometry = load(root, "robustness/geometry_summary.json")
    records = load(root, "robustness/compact/geometry.json")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    paired = [r for r in records if r["near_error_px"] is not None]
    axes[0].scatter(
        [r["reference"]["centers"][0] for r in paired],
        [r["prediction"]["centers"][0] for r in paired],
        s=8,
        alpha=0.5,
        color=colors[0],
    )
    axes[0].plot([0, 1], [0, 1], "--", color="#777777")
    axes[0].set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Reference-mask corridor center / width",
        ylabel="Predicted-mask corridor center / width",
        title=f"Near-anchor valid pairs (n={len(paired)})",
    )
    for i, anchor in enumerate(["near", "far"]):
        selected = [
            next(r for r in geometry if r["model"] == name and r["anchor"] == anchor)
            for name in names
        ]
        axes[1].bar(
            np.arange(3) + (i - 0.5) * 0.35,
            [
                np.nan
                if r["mean_absolute_error_px"] is None
                else r["mean_absolute_error_px"]
                for r in selected
            ],
            width=0.35,
            label=anchor,
            color=colors[i],
        )
    axes[1].set_xticks(np.arange(3), short, rotation=20, ha="right")
    axes[1].set(
        ylabel="Conditional center MAE (96×224 pixels)",
        title="Paired-valid geometry error; see coverage",
    )
    axes[1].legend()
    example = root / "robustness/compact/geometry_example.npz"
    if example.exists():
        with np.load(example, allow_pickle=False) as data:
            target = data["target"]
            axes[2].imshow(target, cmap="gray", vmin=0, vmax=1)
            for i, y in enumerate([0.85, 0.60]):
                row = y * (target.shape[0] - 1)
                axes[2].axhline(row, color="#db8126", ls="--", lw=0.8)
                axes[2].scatter(
                    data["reference_centers"][i] * (target.shape[1] - 1),
                    row,
                    marker="o",
                    s=55,
                    facecolors="none",
                    edgecolors="#39be78",
                    label="Reference" if i == 0 else None,
                )
                axes[2].scatter(
                    data["predicted_centers"][i] * (target.shape[1] - 1),
                    row,
                    marker="x",
                    s=30,
                    color="#3b98db",
                    label="Prediction" if i == 0 else None,
                )
        axes[2].legend(fontsize=7)
        axes[2].set(
            title="First paired-valid corridor example",
            xlabel="Evaluation-image x (pixels)",
            ylabel="Evaluation-image y (pixels)",
        )
    finish(fig, "geometry")
    with np.load(
        root / "robustness/compact/failure_cases.npz", allow_pickle=False
    ) as data:
        probabilities = data["probabilities"]
        targets = data["targets"]
    cases = load(root, "robustness/compact/failure_cases.json")
    threshold = load(root, "robustness/compact/results.json")["conditions"][0][
        "threshold"
    ]
    fig, axes = plt.subplots(
        len(cases), 4, figsize=(11, 6), layout="constrained", squeeze=False
    )
    for row, (p, target, case) in enumerate(zip(probabilities, targets, cases)):
        pred = p > threshold
        rgb = np.zeros((*p.shape, 3))
        rgb[pred & target] = [0.15, 0.7, 0.3]
        rgb[pred & ~target] = [0.2, 0.4, 0.9]
        rgb[~pred & target] = [0.85, 0.2, 0.2]
        q = np.clip(p, 1e-7, 1 - 1e-7)
        entropy = -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
        axes[row, 0].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(rgb)
        heat = axes[row, 3].imshow(entropy, cmap="magma", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f'{case["id"]}\nF1={case["f1"]:.3f}', fontsize=8)
        for col, ax in enumerate(axes[row]):
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(
                    [
                        "Reference mask",
                        "Prediction",
                        "TP green / FP blue / FN red",
                        "Predictive entropy (bits)",
                    ][col],
                    fontsize=9,
                )
    fig.colorbar(
        heat, ax=axes[:, 3], label="Bernoulli entropy (bits)", fraction=0.035, pad=0.02
    )
    finish(fig, "failure_cases")


def write_report(root):
    root = Path(root)
    kalman = load(root, "kalman/kalman_results.json")
    eff = load(root, "efficiency/efficiency.json")["models"]
    nominal = [r for r in kalman["verification_scenarios"] if not r["stress"]]
    stress = [r for r in kalman["verification_scenarios"] if r["stress"]]
    mean = lambda rows, key: float(np.mean([r[key] for r in rows]))
    count = len(nominal)
    vstart = kalman["verification_seed_start"]
    vrange = f"{vstart}–{vstart+count-1}"
    initial = load(root, "kalman/initial_stress_results.json")
    intermediate = load(root, "kalman/reacquisition_only_results.json")
    mid_forced = intermediate["forced_initialization_verification"]
    mid_start = intermediate["verification_seed_start"]
    mid_range = f"{mid_start}–{mid_start+len(mid_forced)-1}"
    old_error = initial["diagnostic_seed_24"]["filtered_rmse_observed"]
    lines = [
        "# Additional experiments: filtering, robustness and diagnostics",
        "",
        "Original verified outputs remain in `results/via` and `results/analysis`; these new experiments are separate. None of the VIA filenames are treated as consecutive driving frames.",
        "",
        "## 1. Synthetic nominal filtering",
        "",
        f"Two analytic normalized-image corridor centers are observed at 10 Hz for 240 steps, with Gaussian measurement standard deviation 0.012. Development noise seeds 7–{6+count} and separate verification seeds {vrange} are evaluated. Summary errors below use the verification set and exclude the first 10 steps. Q/R are specified by the simulation; the wrapper recovery rule was fixed before verification.",
        "",
        f"Mean observed-frame RMSE: raw **{mean(nominal,'raw_rmse_observed'):.6f}**, Kalman **{mean(nominal,'filtered_rmse_observed'):.6f}**, EMA **{mean(nominal,'ema_rmse_observed'):.6f}**, in normalized image width.",
        "",
        "![Synthetic nominal filter](figures/kalman_nominal.png)",
        "",
        "*First two panels: known analytic truth, noisy observations, prior predictions, posterior corrections and posterior ±1.96σ for seed 7. Third panel: actual trained-model measurements and filtered estimates from the controlled image sequence. Initial estimates are withheld during the five-observation startup window. All three panels are synthetic, not vehicle motion measurements.*",
        "",
        "## 2. Synthetic temporal stress and parameter sensitivity",
        "",
        f"The stress case adds a 25-frame contiguous gap, random missing observations, and injected outliers. Across the {count} verification noise seeds, observed-frame RMSE is **{mean(stress,'raw_rmse_observed'):.6f}** raw and **{mean(stress,'filtered_rmse_observed'):.6f}** filtered. Initialized-frame filtered RMSE, including gaps, is **{mean(stress,'filtered_rmse_all'):.6f}**. Mean nominal posterior interval coverage is **{100*mean(nominal,'position_95_interval_coverage'):.2f}%**; coverage is diagnostic, not a guarantee.",
        "",
        "![Synthetic stress](figures/kalman_stress.png)",
        "",
        f"*Seed-7 stress trajectory, mean ± SD observed-frame RMSE across {count} verification noise seeds, and a predeclared three-value acceleration-noise sweep. EMA uses α=0.25; filtering is causal.*",
        "",
        f"During development, a first-observation outlier in seed 24 produced gate lock: valid updates were rejected after a biased velocity estimate. Its original observed-frame filter RMSE was {old_error:.6f} normalized widths. The failed outputs are retained as `kalman/initial_stress_results.json` and `kalman/initial_failure_trajectory.json`. The wrapper now initializes only after five coherent complete observations (at least 0.4 s at 10 Hz) and reacquires from the median of eight consecutive rejected, full observations only when each anchor range is at most 0.06; it resets velocity and covariance explicitly. This heuristic is separate from the linear Kalman update and can still fail on coherent false boundaries.",
        "",
        f"The final stable-initialization seed-24 RMSE is {kalman['diagnostic_seed_24']['filtered_rmse_observed']:.6f}. Independent verification seeds {vrange} were evaluated after the recovery rule was fixed. The failed run can be reproduced with `initialization_observations=1` and `reacquire_after=None`.",
        "",
        f"The intermediate reacquisition-only variant is retained in `kalman/reacquisition_only_results.json`. On its additional forced-initial-outlier verification using seeds {mid_range}, mean filtered RMSE was {mean(mid_forced,'filtered_rmse_observed'):.6f} versus {mean(mid_forced,'raw_rmse_observed'):.6f} raw: recovery alone did not fix startup bias. Stable-window initialization was added before the final verification seeds {vrange} were evaluated.",
        "",
        f"On the final forced-initialization verification, mean filtered RMSE is {np.mean([r['filtered_rmse_observed'] for r in kalman['forced_initialization_verification']]):.6f}. Availability counts are saved: frames without a valid initialized state are not filled or treated as accurate predictions.",
        "",
        "A separate 48-frame controlled render executes the complete trained-model → measurement → filter pipeline. See `temporal/sequence.json`; it is explicitly `synthetic_controlled`.",
        "",
        f"Controlled-image matched-frame RMSE is {load(root,'temporal/sequence_metrics.json')['raw_rmse']:.6f} raw versus {load(root,'temporal/sequence_metrics.json')['filtered_rmse']:.6f} filtered, with {load(root,'temporal/sequence_metrics.json')['comparable_frames']} scored frames out of 48. The fixed simulation noise settings are not calibrated to this renderer or model; smoothing can introduce lag and need not improve image-derived accuracy.",
        "",
        "## 3. New fog levels and restoration comparison",
        "",
        "Frozen compact, no-restoration and U-Net checkpoints are evaluated on all 610 held-out miniature-track images. New atmospheric β levels are 0.15, 0.45, 1.8 and 2.1; β=0 is repeated solely to verify the original confusion matrices and provide a reference. Thresholds are unchanged and not selected from test data.",
        "",
        "| Model | β=0 F1 (%) | β=0.15 | β=0.45 | β=1.8 | β=2.1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ["compact", "compact_no_defog", "unet"]:
        c = load(root, f"robustness/{name}/results.json")["conditions"]
        lines.append(
            "| " + name + " | " + " | ".join(f'{100*r["f1"]:.2f}' for r in c) + " |"
        )
    lines += [
        "",
        "![Extended fog](figures/fog_robustness.png)",
        "",
        "*Pixel F1 and restoration-enabled minus disabled differences. Independent training initialization/augmentation histories and the restoration pretraining budget prevent a causal claim that restoration alone explains differences.*",
        "",
        "## 4. Controlled CPU efficiency",
        "",
        f"Models are benchmarked serially in fresh subprocesses on an identical SHA-256-verified synthetic 96×224 RGB probe, batch 1, two CPU threads, {eff[0]['warmup']} warmups, and {eff[0]['repeats']}×{eff[0]['iterations_per_repeat']} timed forwards. Input generation, loading and postprocessing are excluded. Memory polling is sampled total resident process memory, not activation-only or GPU memory.",
        "",
        "| Model | Parameters | Median forward ms | P95 ms | Sampled process RSS MiB |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in eff:
        lines.append(
            f"| {row['model']} | {row['parameters']:,} | {row['latency_median_ms']:.3f} | {row['latency_p95_ms']:.3f} | {row['sampled_peak_inference_rss_mb']:.1f} |"
        )
    lines += [
        "",
        "![Efficiency](figures/efficiency.png)",
        "",
        "*Actual timing distributions, parameter counts and sampled process resident memory. These observations are machine-specific, not real-time vehicle-deployment claims.*",
        "",
        "## 5. Confidence and error diagnostics",
        "",
        "Calibration uses probability assigned to the class selected by the frozen validation threshold. Fifteen equal-width bins produce weighted ECE; Brier/NLL use all pixels. Class-conditional Brier errors expose background imbalance. Predictive entropy is not Bayesian or epistemic uncertainty.",
        "",
        "| Model | Brier | NLL | ECE | Lane Brier | Background Brier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ["compact", "compact_no_defog", "unet"]:
        c = next(
            r
            for r in load(root, f"robustness/{name}/calibration.json")
            if r["beta"] == 0
        )
        lines.append(
            f"| {name} | {c['brier']:.5f} | {c['negative_log_likelihood']:.5f} | {c['ece']:.5f} | {c['positive_brier']:.5f} | {c['negative_brier']:.5f} |"
        )
    lines += [
        "",
        "![Confidence](figures/confidence.png)",
        "",
        "*Selected-class reliability and true-class-conditional squared probability errors; the population is background-heavy.*",
        "",
        "![Failure cases](figures/failure_cases.png)",
        "",
        "*The three lowest finite per-image F1 scores for the compact checkpoint on original held-out images, with ID tie-breaking. This is an explicit worst-case selection rule. Color errors and entropy are computed from the saved predictions. Source photographs are excluded.*",
        "",
        "## 6. Image-derived lane geometry",
        "",
        "Near/far row bands at y=0.85/0.60 extract the nearest narrow foreground runs bracketing the camera center. Missing, merged or unbracketed markings produce no estimate. Reference geometry is extracted from annotated masks using the same rule; it is a mask-consistency proxy, not calibrated ego-lane ground truth. Errors are conditional on both reference and predicted centers being available; coverage is reported explicitly.",
        "",
        "| Model | Anchor | Valid pairs / reference available | Conditional MAE (evaluation pixels) |",
        "|---|---|---:|---:|",
    ]
    for r in load(root, "robustness/geometry_summary.json"):
        mae = (
            "undefined"
            if r["mean_absolute_error_px"] is None
            else f"{r['mean_absolute_error_px']:.3f}"
        )
        lines.append(
            f"| {r['model']} | {r['anchor']} | {r['paired_observations']} / {r['reference_available']} | {mae} |"
        )
    lines += [
        "",
        "![Geometry](figures/geometry.png)",
        "",
        "*Paired center scatter, conditional mean errors, and the first paired-valid compact example. Coverage and failures limit interpretation; unknown centers are never filled using annotations.*",
        "",
        "## Reproduce",
        "",
        "From the repository root, after installing the environment and preparing `data/via`:",
        "",
        "```bash",
        "python -m rmldnet.extensions all --output runs/extensions --full-checkpoint runs/via/swin_b/best.pt",
        "```",
        "",
        "The full Swin-B trained checkpoint is regenerated using the existing training instructions. Individual commands `kalman`, `robustness`, `efficiency`, and `figures` are also supported. No original result files are overwritten by these commands.",
        "",
        "The additional segmentation bundle is compatible with the existing transactional registry. New filtering/geometry diagnostics remain explicit JSON/CSV records rather than being mislabeled as pixel-confusion experiments. PNG, SVG and PDF exports share the same computed data.",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n")
