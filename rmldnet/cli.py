"""Training, paired restoration, evaluation, inference and measured latency."""

import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from .data import LaneDataset, synthetic
from .model import RMLDNet
from .losses import objective
from .metrics import Confusion
from .tracking import LaneTracker, decode_lanes, lane_state, state_points


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def build(config, device):
    return RMLDNet(**config["model"]).to(device)


def save(path, model, config, epoch, optimizer=None, scheduler=None):
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "torch_version": str(torch.__version__),
        },
        path,
    )


def load(path, device):
    c = torch.load(path, map_location=device, weights_only=True)
    model = build(c["config"], device)
    model.load_state_dict(c["model"])
    return model, c


def evaluate(model, loader, device):
    model.eval()
    metric = Confusion()
    losses = []
    with torch.inference_mode():
        for b in loader:
            out = model(b["image"].to(device))
            loss, _ = objective(
                out, b["mask"].to(device), b["instance_valid"].to(device)
            )
            losses.append(loss.item())
            metric.update(
                out["logits"].sigmoid().cpu().numpy()[:, 0] > 0.5, b["mask"].numpy() > 0
            )
    return dict(
        metric.compute(),
        loss=float(np.mean(losses)),
        samples=len(loader.dataset),
        tracking=False,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "command",
        choices=[
            "synthetic",
            "train",
            "evaluate",
            "infer",
            "pretrain-defog",
            "benchmark",
        ],
    )
    p.add_argument("--config", default="configs/smoke.yaml")
    p.add_argument("--manifest")
    p.add_argument("--val-manifest")
    p.add_argument("--output", default="runs/experiment")
    p.add_argument("--checkpoint")
    p.add_argument("--encoder-weights")
    p.add_argument("--defog-weights")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--count", type=int, default=12)
    p.add_argument("--images", nargs="+")
    p.add_argument("--track", action="store_true")
    p.add_argument("--steps", type=int, default=20)
    a = p.parse_args()
    seed_all(a.seed)
    torch.set_num_threads(2)
    outdir = Path(a.output)
    outdir.mkdir(parents=True, exist_ok=True)
    if a.command == "synthetic":
        synthetic(outdir, a.count, a.seed)
        return
    config = yaml.safe_load(Path(a.config).read_text())
    if a.checkpoint:
        model, checkpoint = load(a.checkpoint, a.device)
        config = checkpoint["config"]
    else:
        model = build(config, a.device)
    if a.encoder_weights:
        model.load_encoder(a.encoder_weights)
    if a.defog_weights:
        model.defogger.load_state_dict(
            torch.load(a.defog_weights, map_location=a.device, weights_only=True)
        )
    size = tuple(config["size"])
    epochs = config["epochs"]
    if a.command in ("train", "pretrain-defog", "evaluate"):
        if not a.manifest:
            p.error("--manifest required")
        dataset = LaneDataset(a.manifest, size, a.command != "evaluate")
        loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=a.command != "evaluate",
            num_workers=0,
        )
    if a.command in ("train", "pretrain-defog"):
        if a.command == "train" and not a.val_manifest:
            p.error("--val-manifest required for training")
        if a.command == "pretrain-defog" and model.defogger is None:
            p.error("defogging must be enabled")
        if a.command == "train":
            train_paths = {
                (dataset.path.parent / r["image"]).resolve() for r in dataset.rows
            }
            valset = LaneDataset(a.val_manifest, size)
            if train_paths & {
                (valset.path.parent / r["image"]).resolve() for r in valset.rows
            }:
                raise ValueError("Train/validation image overlap")
            val = DataLoader(valset, batch_size=config["batch_size"])
        params = (
            model.parameters() if a.command == "train" else model.defogger.parameters()
        )
        opt = torch.optim.AdamW(
            params, lr=config["lr"], weight_decay=config["weight_decay"]
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        history = []
        for epoch in range(epochs):
            model.train()
            total = 0
            for b in loader:
                image = b["image"].to(a.device)
                opt.zero_grad()
                if a.command == "pretrain-defog":
                    # Dataset provides paired synthetic fog/clear supervision; no real clear targets invented.
                    restored, _, _ = model.defogger(image)
                    loss = torch.nn.functional.l1_loss(
                        restored, b["clear"].to(a.device)
                    )
                else:
                    loss, _ = objective(
                        model(image),
                        b["mask"].to(a.device),
                        b["instance_valid"].to(a.device),
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                total += loss.item()
            sched.step()
            row = {"epoch": epoch + 1, "train_loss": total / len(loader)}
            if a.command == "train":
                row["validation"] = evaluate(model, val, a.device)
            history.append(row)
            print(json.dumps(row), flush=True)
            save(outdir / "last.pt", model, config, epoch + 1, opt, sched)
        (outdir / "history.json").write_text(json.dumps(history, indent=2))
        (outdir / "config.yaml").write_text(yaml.safe_dump(config))
        if a.command == "pretrain-defog":
            torch.save(model.defogger.state_dict(), outdir / "defog.pt")
    elif a.command == "evaluate":
        if not a.checkpoint:
            p.error("--checkpoint required")
        result = evaluate(model, loader, a.device)
        (outdir / "metrics.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
    elif a.command == "infer":
        if not a.checkpoint or not a.images:
            p.error("--checkpoint and --images required")
        tracker = LaneTracker()
        results = []
        model.eval()
        for index, name in enumerate(a.images):
            im = Image.open(name).convert("RGB").resize((size[1], size[0]))
            image = (
                torch.from_numpy(np.asarray(im, dtype=np.float32) / 255)
                .permute(2, 0, 1)[None]
                .to(a.device)
            )
            with torch.inference_mode():
                pred = model(image)
            prob = pred["logits"][0, 0].sigmoid().cpu().numpy()
            lanes = decode_lanes(prob, pred["embeddings"][0].cpu().numpy())
            tracks = tracker.update([lane_state(l) for l in lanes]) if a.track else []
            Image.fromarray((prob * 255).astype("uint8")).save(
                outdir / f"{index:06d}_prob.png"
            )
            results.append(
                {
                    "image": name,
                    "lanes_normalized": [l.tolist() for l in lanes],
                    "tracks": [
                        dict(
                            id=t["id"],
                            state=t["state"].tolist(),
                            predicted=t["predicted"],
                            points=state_points(
                                t["state"], np.linspace(0.25, 1, 30)
                            ).tolist(),
                        )
                        for t in tracks
                    ],
                }
            )
        (outdir / "predictions.json").write_text(json.dumps(results, indent=2))
    elif a.command == "benchmark":
        if a.steps < 1:
            p.error("--steps must be positive")
        image = torch.rand(1, 3, *size, device=a.device)
        model.eval()

        def sync():
            if str(a.device).startswith("cuda"):
                torch.cuda.synchronize()

        with torch.inference_mode():
            for _ in range(3):
                model(image)
            sync()
            start = time.perf_counter()
            for _ in range(a.steps):
                model(image)
            sync()
            elapsed = time.perf_counter() - start
        result = {
            "fps": a.steps / elapsed,
            "steps": a.steps,
            "device": a.device,
            "size": size,
            "scope": "network incl. defog/FPN/head; excludes IO, clustering and tracking",
            "parameters": sum(x.numel() for x in model.parameters()),
        }
        (outdir / "benchmark.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))


if __name__ == "__main__":
    main()
