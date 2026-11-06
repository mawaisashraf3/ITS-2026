"""Isolated CPU forward benchmarks with raw timings and sampled resident memory."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import numpy as np
import torch
from .experiment import make_model
from .public_data import digest


def resident_mb():
    pages = int(Path("/proc/self/statm").read_text().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / 1024**2


def worker(model_name, checkpoint, iterations=30, repeats=3):
    if iterations < 1 or repeats < 1:
        raise ValueError("Need positive benchmark iteration counts")
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = make_model(model_name)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["state"])
    del state
    gc.collect()
    model.eval()
    probe_generator = torch.Generator().manual_seed(7)
    image = torch.rand(1, 3, 96, 224, generator=probe_generator)
    timings = []
    with torch.inference_mode():
        for _ in range(5):
            model(image)
        baseline = resident_mb()
        memory = [baseline]
        stop = threading.Event()

        def sample():
            while not stop.wait(0.001):
                memory.append(resident_mb())

        poller = threading.Thread(target=sample, daemon=True)
        poller.start()
        try:
            for repeat in range(repeats):
                for index in range(iterations):
                    start = time.perf_counter_ns()
                    out = model(image)
                    timings.append(
                        {
                            "repeat": repeat,
                            "iteration": index,
                            "milliseconds": (time.perf_counter_ns() - start) / 1e6,
                        }
                    )
                    if not torch.isfinite(out["logits"]).all():
                        raise FloatingPointError("Nonfinite benchmark logits")
        finally:
            stop.set()
            poller.join()
    values = np.array([t["milliseconds"] for t in timings])
    return {
        "model": model_name,
        "device": "cpu",
        "threads": 2,
        "batch_size": 1,
        "size": [96, 224],
        "input_tensor_sha256": hashlib.sha256(image.numpy().tobytes()).hexdigest(),
        "input_kind": "fixed seeded uniform RGB microbenchmark probe",
        "memory_units": "MiB (1024**2 bytes)",
        "warmup": 5,
        "iterations_per_repeat": iterations,
        "repeats": repeats,
        "latency_median_ms": float(np.median(values)),
        "latency_p95_ms": float(np.quantile(values, 0.95)),
        "median_forward_fps": float(1000 / np.median(values)),
        "parameters": sum(p.numel() for p in model.parameters()),
        "parameter_bytes": sum(
            p.numel() * p.element_size() for p in model.parameters()
        ),
        "resident_before_inference_mb": baseline,
        "sampled_peak_inference_rss_mb": max(memory),
        "sampled_rss_increase_mb": max(memory) - baseline,
        "checkpoint_sha256": digest(checkpoint),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "timings": timings,
        "scope": "Network forward including restoration/FPN/head; excludes IO, geometry, tracking and logits-finiteness check. RSS is sampled process resident memory, not allocator or GPU peak.",
    }


def run(bundle, full_checkpoint, output, iterations=30, repeats=3):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ["compact", "compact_no_defog", "unet", "swin_b"]:
        path = (
            Path(full_checkpoint)
            if name == "swin_b"
            else Path(bundle) / name / "best.pt"
        )
        if not path.exists():
            raise FileNotFoundError(f"Benchmark requires trained checkpoint: {path}")
        command = [
            sys.executable,
            "-m",
            "rmldnet.efficiency",
            "--worker",
            name,
            "--checkpoint",
            str(path),
            "--iterations",
            str(iterations),
            "--repeats",
            str(repeats),
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        rows.append(json.loads(completed.stdout))
    report = {
        "execution": "Serial fresh subprocess per model; same 96x224 input, batch 1, two CPU threads",
        "models": rows,
    }
    (output / "efficiency.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker", choices=["compact", "compact_no_defog", "unet", "swin_b"]
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not args.worker or not args.checkpoint:
        parser.error("--worker and --checkpoint required")
    print(
        json.dumps(worker(args.worker, args.checkpoint, args.iterations, args.repeats))
    )


if __name__ == "__main__":
    main()
