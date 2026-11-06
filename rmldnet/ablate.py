"""Generate and optionally run controlled neural ablations on supplied splits."""

import argparse
import copy
from pathlib import Path
import subprocess
import sys
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/smoke.yaml")
    p.add_argument("--manifest", required=True)
    p.add_argument("--val-manifest", required=True)
    p.add_argument("--output", default="runs/ablations")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    base = yaml.safe_load(Path(a.config).read_text())
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=True)
    for name, defog, fpn in [
        ("full", True, True),
        ("no_defog", False, True),
        ("no_fpn", True, False),
    ]:
        config = copy.deepcopy(base)
        config["model"]["defog"] = defog
        config["model"]["fpn"] = fpn
        path = root / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config))
        for seed in a.seeds:
            cmd = [
                sys.executable,
                "-m",
                "rmldnet.cli",
                "train",
                "--config",
                str(path),
                "--manifest",
                a.manifest,
                "--val-manifest",
                a.val_manifest,
                "--device",
                a.device,
                "--seed",
                str(seed),
                "--output",
                str(root / f"{name}_{seed}"),
            ]
            print(" ".join(cmd), flush=True)
            if a.execute:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
