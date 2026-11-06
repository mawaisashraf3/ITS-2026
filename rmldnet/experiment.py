"""Reproducible measured-data study with validation-only model/threshold selection."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from .baselines import UNet
from .cli import seed_all
from .data import LaneDataset
from .losses import objective
from .metrics import Confusion
from .model import RMLDNet
from .public_data import digest


def make_model(name, encoder=None):
    if name == "unet":
        return UNet()
    model = RMLDNet(
        "swin_b" if name == "swin_b" else "smoke",
        128 if name == "swin_b" else 32,
        defog=name != "compact_no_defog",
    )
    if name == "swin_b" and encoder:
        model.load_encoder(encoder)
    return model


def perturb(image, beta):
    if not beta:
        return image
    depth = torch.linspace(0.4, 1.8, image.shape[-2], device=image.device)[
        None, None, :, None
    ]
    t = torch.exp(-beta * depth)
    return image * t + 0.85 * (1 - t)


def measure(model, loader, beta=0, threshold=0.5, collect=False):
    model.eval()
    metric = Confusion()
    per_image = []
    cache = []
    start = time.perf_counter()
    with torch.inference_mode():
        for b in loader:
            prob = model(perturb(b["image"], beta))["logits"].sigmoid()[:, 0].numpy()
            target = b["mask"].numpy() > 0
            metric.update(prob > threshold, target)
            for p, t in zip(prob, target):
                m = Confusion()
                m.update(p > threshold, t)
                per_image.append(m.compute())
            if collect:
                cache.append((prob, target))
    return dict(
        metric.compute(),
        samples=len(loader.dataset),
        elapsed_seconds=time.perf_counter() - start,
        threshold=threshold,
        beta=beta,
        per_image=per_image,
    ), cache


def choose_threshold(cache):
    scores = []
    for t in np.arange(0.1, 0.91, 0.05):
        m = Confusion()
        for prob, target in cache:
            m.update(prob > t, target)
        scores.append((m.compute()["f1"] or 0, float(t)))
    return max(scores, key=lambda x: (x[0], -abs(x[1] - 0.5)))[1]


def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def train(model, train_loader, val_loader, epochs, lr, output):
    # Lower learning rate for pretrained encoder; newly initialized heads/restoration use lr.
    if isinstance(model, RMLDNet) and model.encoder[0][0].out_channels == 128:
        enc = list(model.encoder.parameters())
        enc_ids = {id(p) for p in enc}
        opt = torch.optim.AdamW(
            [
                {"params": enc, "lr": lr * 0.01},
                {
                    "params": [p for p in model.parameters() if id(p) not in enc_ids],
                    "lr": lr,
                },
            ],
            weight_decay=0.01,
        )
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best = -1
    history = []
    best_state = None
    best_threshold = 0.5
    for epoch in range(epochs):
        start = time.perf_counter()
        model.train()
        loss_sum = 0
        samples = 0
        for b in train_loader:
            opt.zero_grad()
            prediction = model(b["image"])
            loss, _ = objective(prediction, b["mask"], b["instance_valid"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            loss_sum += loss.item() * len(b["image"])
            samples += len(b["image"])
        sched.step()
        _, cache = measure(model, val_loader, collect=True)
        threshold = choose_threshold(cache)
        metric = Confusion()
        for prob, target in cache:
            metric.update(prob > threshold, target)
        score = metric.compute()["f1"] or 0
        row = {
            "epoch": epoch + 1,
            "loss": loss_sum / samples,
            "validation_f1": score,
            "threshold": threshold,
            "seconds": time.perf_counter() - start,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        if score > best:
            best = score
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            best_threshold = threshold
            torch.save(
                {"state": best_state, "threshold": best_threshold, "epoch": epoch + 1},
                output / "best.pt",
            )
    model.load_state_dict(best_state)
    return history, best_threshold


def write_svg(path, history):
    pts = " ".join(
        f'{60+i*520/max(1,len(history)-1):.1f},{260-220*r["validation_f1"]:.1f}'
        for i, r in enumerate(history)
    )
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="310"><rect width="640" height="310" fill="white"/><text x="50" y="25" font-size="18">Validation pixel F1 by epoch</text><path d="M60 40 V260 H590" fill="none" stroke="black"/><polyline points="{pts}" fill="none" stroke="#1466b8" stroke-width="3"/><text x="60" y="290">Epoch 1</text><text x="520" y="290">Epoch {len(history)}</text><text x="25" y="45">1.0</text><text x="25" y="265">0.0</text></svg>'
    )


def visualize_masks(model, dataset, threshold, path):
    # Derived masks only: no source photographs redistributed.
    n = min(6, len(dataset))
    h, w = dataset.size
    canvas = Image.new("RGB", (w * 3, h * n), "white")
    model.eval()
    with torch.inference_mode():
        for i in range(n):
            b = dataset[i]
            prob = model(b["image"][None])["logits"][0, 0].sigmoid().numpy()
            gt = b["mask"].numpy() > 0
            pred = prob > threshold
            diff = np.zeros((h, w, 3), dtype=np.uint8)
            diff[gt & pred] = [40, 180, 80]
            diff[gt & ~pred] = [210, 50, 50]
            diff[~gt & pred] = [50, 100, 230]
            for j, arr in enumerate(
                [
                    np.repeat((gt * 255).astype("uint8")[:, :, None], 3, 2),
                    np.repeat((pred * 255).astype("uint8")[:, :, None], 3, 2),
                    diff,
                ]
            ):
                canvas.paste(Image.fromarray(arr), (w * j, h * i))
    canvas.save(path)


def write_results(output, summary, sample_ids):
    """Store readable aggregate JSON and indexed raw per-image JSONL separately."""
    output = Path(output)
    raw = output / "per_image_metrics.jsonl"
    with raw.open("w") as stream:
        for condition in summary["conditions"]:
            per_image = condition.pop("per_image")
            if len(per_image) != len(sample_ids):
                raise ValueError("Per-image result count differs from test manifest")
            for sample_id, metrics in zip(sample_ids, per_image):
                stream.write(
                    json.dumps(
                        {"id": sample_id, "beta": condition["beta"], "metrics": metrics}
                    )
                    + "\n"
                )
    summary["per_image_metrics_file"] = raw.name
    (output / "results.json").write_text(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/via")
    p.add_argument("--output", default="runs/via")
    p.add_argument(
        "--model",
        choices=["compact", "compact_no_defog", "swin_b", "unet"],
        default="compact",
    )
    p.add_argument("--encoder")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--size", type=int, nargs=2, default=[96, 224])
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument("--defog-pretrain-epochs", type=int, default=2)
    a = p.parse_args()
    seed_all(a.seed)
    torch.set_num_threads(2)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(a.data)
    datasets = {
        s: LaneDataset(root / f"{s}.jsonl", a.size, s == "train")
        for s in ["train", "val", "test"]
    }
    hashes = [{r["sha256"] for r in datasets[s].rows} for s in ["train", "val", "test"]]
    assert not (
        hashes[0] & hashes[1] or hashes[0] & hashes[2] or hashes[1] & hashes[2]
    ), "Split content leakage"
    assert not (
        {r["group"] for r in datasets["train"].rows}
        & {r["group"] for r in datasets["val"].rows}
    )
    loaders = {
        s: DataLoader(ds, batch_size=a.batch_size, shuffle=s == "train", num_workers=0)
        for s, ds in datasets.items()
    }
    model = make_model(a.model, a.encoder)
    config = vars(a).copy()
    config.update(
        python=platform.python_version(),
        torch=str(torch.__version__),
        device="cpu",
        parameters=sum(p.numel() for p in model.parameters()),
        encoder_sha256=digest(a.encoder) if a.encoder else None,
        encoder_pretraining="torchvision ImageNet-1K (verified official weights)"
        if a.encoder
        else "random",
        instance_supervision=False,
        split_sha256={s: digest(root / f"{s}.jsonl") for s in datasets},
    )
    if not a.evaluate_only:
        (out / "config.json").write_text(json.dumps(config, indent=2))
    else:
        saved = json.loads((out / "config.json").read_text())
        if saved["model"] != a.model or saved["size"] != a.size:
            raise ValueError("Evaluation model/size differ from saved experiment")
        if saved["split_sha256"] != config["split_sha256"]:
            raise ValueError("Evaluation split hashes differ from saved experiment")
    if a.evaluate_only:
        checkpoint = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state"])
        threshold = checkpoint["threshold"]
        history = json.loads((out / "history.json").read_text())
    else:
        restoration_history = []
        if isinstance(model, RMLDNet) and model.defogger is not None:
            opt = torch.optim.AdamW(
                model.defogger.parameters(), lr=a.lr, weight_decay=0.01
            )
            for epoch in range(a.defog_pretrain_epochs):
                total = 0
                count = 0
                for b in loaders["train"]:
                    opt.zero_grad()
                    restored, _, _ = model.defogger(b["image"])
                    loss = torch.nn.functional.l1_loss(restored, b["clear"])
                    loss.backward()
                    opt.step()
                    total += loss.item() * len(b["image"])
                    count += len(b["image"])
                restoration_history.append(
                    {"epoch": epoch + 1, "paired_synthetic_l1": total / count}
                )
            (out / "restoration_history.json").write_text(
                json.dumps(restoration_history, indent=2)
            )
        history, threshold = train(
            model, loaders["train"], loaders["val"], a.epochs, a.lr, out
        )
    # Test is evaluated only after validation has selected model and threshold.
    results = []
    for beta in [0.0, 0.7, 1.2]:
        result, _ = measure(model, loaders["test"], beta, threshold)
        results.append(result)
        print(
            json.dumps({k: v for k, v in result.items() if k != "per_image"}),
            flush=True,
        )
    summary = {
        "model": a.model,
        "seed": a.seed,
        "selection": "best validation pixel F1; threshold validation-only",
        "checkpoint_tensor_sha256": state_hash(model.state_dict()),
        "conditions": results,
    }
    write_results(out, summary, [row["id"] for row in datasets["test"].rows])
    write_svg(out / "learning.svg", history)
    visualize_masks(model, datasets["test"], threshold, out / "masks.png")
    with (out / "metrics.csv").open("w") as f:
        keys = [
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
        writer = csv.DictWriter(
            f, fieldnames=keys, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
