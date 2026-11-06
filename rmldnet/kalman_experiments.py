"""Seeded numerical lane-trajectory experiments; all outputs explicitly synthetic."""

from dataclasses import asdict, replace
import csv
import json
from pathlib import Path
import numpy as np
from .kalman import LaneFilterConfig, LanePositionFilter


def synthetic_trajectory(seed=7, stress=False, count=240, force_initial_outlier=False):
    rng = np.random.default_rng(seed)
    time = np.arange(count) * 0.1
    truth = np.column_stack(
        [0.50 + 0.06 * np.sin(0.30 * time), 0.48 + 0.04 * np.sin(0.30 * time + 0.2)]
    )
    measured = truth + rng.normal(0, 0.012, truth.shape)
    outliers = np.zeros(count, dtype=bool)
    missing = np.zeros(count, dtype=bool)
    if stress:
        missing[75:100] = True
        missing |= rng.random(count) < 0.06
        missing[:3] = False
        outliers[145:155] = True
        outliers |= rng.random(count) < 0.025
        outliers[missing] = False
        measured[outliers] += rng.choice([-1, 1], (outliers.sum(), 1)) * 0.12
        measured[missing] = np.nan
    if force_initial_outlier:
        measured[0] = truth[0] + np.array([0.12, 0.12])
        missing[0] = False
        outliers[0] = True
    return time, truth, measured, missing, outliers


def evaluate(seed=7, stress=False, config=None, force_initial_outlier=False):
    config = config or LaneFilterConfig()
    time, truth, measured, missing, outliers = synthetic_trajectory(
        seed, stress, force_initial_outlier=force_initial_outlier
    )
    tracker = LanePositionFilter(config)
    records = []
    ema = None
    ema_values = []
    filtered = []
    stds = []
    accepted = []
    for i, (t, z) in enumerate(zip(time, measured)):
        raw = [float(v) if np.isfinite(v) else None for v in z]
        estimate = tracker.step(raw, float(t))
        if np.isfinite(z).all():
            ema = z.copy() if ema is None else 0.25 * z + 0.75 * ema
        ema_values.append(ema.copy())
        filtered.append(
            [np.nan, np.nan] if estimate["state"] is None else estimate["state"][:2]
        )
        stds.append(
            [np.nan, np.nan]
            if estimate["covariance"] is None
            else np.sqrt(np.diag(estimate["covariance"])[:2])
        )
        accepted.append(estimate["accepted"])
        records.append(
            {
                "time_s": float(t),
                "truth": truth[i].tolist(),
                "measurement": raw,
                "missing": bool(missing[i]),
                "outlier": bool(outliers[i]),
                **estimate,
            }
        )
    filtered = np.asarray(filtered)
    ema_values = np.asarray(ema_values)
    stds = np.asarray(stds)
    use = np.arange(len(time)) >= 10
    available = use & np.isfinite(filtered).all(axis=1)
    observed = available & ~missing

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    raw_second = np.diff(measured - truth, n=2, axis=0)
    filtered_second = np.diff(filtered - truth, n=2, axis=0)
    triples = observed[:-2] & observed[1:-1] & observed[2:]
    summary = {
        "seed": seed,
        "source_kind": "synthetic_analytic_trajectory",
        "stress": stress,
        "forced_initial_outlier": force_initial_outlier,
        "frames": len(time),
        "burn_in": 10,
        "initialized_scored_frames": int(available.sum()),
        "frames_after_burn_in": int(use.sum()),
        "raw_rmse_observed": rmse(measured[observed], truth[observed]),
        "filtered_rmse_observed": rmse(filtered[observed], truth[observed]),
        "filtered_rmse_all": rmse(filtered[available], truth[available]),
        "ema_rmse_all": rmse(ema_values[available], truth[available]),
        "ema_rmse_observed": rmse(ema_values[observed], truth[observed]),
        "raw_error_second_difference_rms_contiguous": float(
            np.sqrt(np.mean(raw_second[triples] ** 2))
        ),
        "filtered_error_second_difference_rms_contiguous": float(
            np.sqrt(np.mean(filtered_second[triples] ** 2))
        ),
        "position_95_interval_coverage": float(
            np.mean(
                np.abs(filtered[available] - truth[available]) <= 1.96 * stds[available]
            )
        ),
        "estimation_error_second_difference_rms": float(
            np.sqrt(
                np.mean(
                    np.diff(filtered[available] - truth[available], n=2, axis=0) ** 2
                )
            )
        ),
        "rejected_observations": int(sum((~missing) & ~np.array(accepted))),
        "reacquisitions": sum(r["status"] == "reacquired" for r in records),
        "missing_frames": int(missing.sum()),
        "outlier_frames": int(outliers.sum()),
        "filter_config": asdict(config),
    }
    return summary, records


def _suite(config, seeds, verification_start=None, trace_output=None):
    scenarios = []
    verification = []
    forced = []
    for stress in [False, True]:
        for seed in range(7, 7 + seeds):
            summary, records = evaluate(seed, stress, config)
            scenarios.append(summary)
            if trace_output is not None and seed == 7:
                name = "stress_trajectory.json" if stress else "nominal_trajectory.json"
                (trace_output / name).write_text(
                    json.dumps(records, indent=2, allow_nan=False)
                )
        if verification_start is not None:
            for seed in range(verification_start, verification_start + seeds):
                summary, _ = evaluate(seed, stress, config)
                verification.append(summary)
    if verification_start is not None:
        for seed in range(verification_start, verification_start + seeds):
            summary, _ = evaluate(seed, True, config, force_initial_outlier=True)
            forced.append(summary)
    diagnostic, trace = evaluate(24, True, config)
    return {
        "source_kind": "synthetic_only",
        "seeds": seeds,
        "development_seed_start": 7,
        "verification_seed_start": verification_start,
        "scenarios": scenarios,
        "verification_scenarios": verification,
        "forced_initialization_verification": forced,
        "diagnostic_seed_24": diagnostic,
    }, trace


def run(output, seeds=20):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    original, failed_trace = _suite(
        LaneFilterConfig(initialization_observations=1, reacquire_after=None), seeds
    )
    original["implementation"] = (
        "single-observation initialization without reacquisition; reproduced retained failed method"
    )
    (output / "initial_stress_results.json").write_text(json.dumps(original, indent=2))
    (output / "initial_failure_trajectory.json").write_text(
        json.dumps(failed_trace, indent=2, allow_nan=False)
    )
    intermediate, _ = _suite(
        LaneFilterConfig(initialization_observations=1), seeds, 107
    )
    intermediate["implementation"] = (
        "single-observation initialization plus reacquisition; retained intermediate method"
    )
    (output / "reacquisition_only_results.json").write_text(
        json.dumps(intermediate, indent=2)
    )
    report, corrected_trace = _suite(LaneFilterConfig(), seeds, 207, output)
    (output / "corrected_failure_trajectory.json").write_text(
        json.dumps(corrected_trace, indent=2, allow_nan=False)
    )
    sensitivity = []
    for variance in [0.00004, 0.0004, 0.004]:
        row, _ = evaluate(
            7, True, replace(LaneFilterConfig(), acceleration_variance=variance)
        )
        sensitivity.append(row)
    report.update(
        process_noise_sensitivity=sensitivity,
        scope="Fixed analytic image-coordinate path, not driving motion. Development seeds exposed initialization/gate lock; intermediate seed-107 challenges exposed startup bias. Final settings use separate verification seeds 207 onward.",
    )
    (output / "kalman_results.json").write_text(json.dumps(report, indent=2))
    keys = [
        "seed",
        "stress",
        "forced_initial_outlier",
        "raw_rmse_observed",
        "filtered_rmse_observed",
        "filtered_rmse_all",
        "ema_rmse_observed",
        "ema_rmse_all",
        "position_95_interval_coverage",
        "rejected_observations",
        "reacquisitions",
        "missing_frames",
        "initialized_scored_frames",
        "frames_after_burn_in",
    ]
    for filename, rows in [
        ("kalman_metrics.csv", report["scenarios"]),
        ("kalman_verification_metrics.csv", report["verification_scenarios"]),
        (
            "initialization_verification_metrics.csv",
            report["forced_initialization_verification"],
        ),
    ]:
        with (output / filename).open("w") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=keys, extrasaction="ignore", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    return report
