"""Execute additional temporal, robustness, confidence and geometry experiments."""

import argparse
import json
import logging
from pathlib import Path
from .efficiency import run as efficiency_run
from .kalman_experiments import run as kalman_run
from .robustness import run as robustness_run
from .temporal import render_sequence, run_sequence
from .storage import ExperimentStore

LOGGER = logging.getLogger(__name__)


def execute(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.command in ("all", "efficiency"):
        LOGGER.info("Running serial isolated CPU benchmarks before other experiments")
        efficiency_run(
            args.bundle,
            args.full_checkpoint,
            output / "efficiency",
            args.iterations,
            args.repeats,
        )
    if args.command in ("all", "kalman"):
        kalman_run(output / "kalman", args.seeds)
        manifest = render_sequence(output / "temporal" / "render")
        run_sequence(manifest, Path(args.bundle) / "compact", output / "temporal")
    if args.command in ("all", "robustness"):
        robustness_run(args.bundle, args.data, output / "robustness")
        with ExperimentStore(output / "experiments.sqlite3") as store:
            original = store.ingest(args.bundle)
            extended = store.ingest(output / "robustness")
            integrity = store.integrity()
            if not integrity["ok"]:
                raise ValueError("Extension registry integrity failure")
            (output / "database_verification.json").write_text(
                json.dumps(
                    {
                        "original": original,
                        "extended": extended,
                        "integrity": integrity,
                    },
                    indent=2,
                )
            )
    if args.command in ("all", "figures"):
        from .extension_figures import create_figures, write_report

        create_figures(output)
        write_report(output)
    LOGGER.info("Completed %s; outputs in %s", args.command, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["all", "kalman", "efficiency", "robustness", "figures"]
    )
    parser.add_argument("--output", default="runs/extensions")
    parser.add_argument("--bundle", default="results/via")
    parser.add_argument("--data", default="data/via")
    parser.add_argument("--full-checkpoint", default="runs/via/swin_b/best.pt")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.seeds < 2 or args.iterations < 1 or args.repeats < 1:
        parser.error("Use >=2 noise seeds and positive benchmark repetitions")
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        execute(args)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
