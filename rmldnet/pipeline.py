"""Execute database ingestion, integrity checking, statistics and optional sensitivity."""

import argparse
from dataclasses import asdict
import json
import logging
import sqlite3
from pathlib import Path
from .settings import Settings
from .storage import ExperimentStore
from .analysis import statistics, sensitivity, figures


def run(settings, with_sensitivity=False):
    output = Path(settings.output)
    output.mkdir(parents=True, exist_ok=True)
    with ExperimentStore(settings.database) as store:
        ingestion = store.ingest(settings.bundle)
        integrity = store.integrity()
        if not integrity["ok"]:
            raise ValueError("Database failed integrity verification")
        metrics = store.metrics()
    result = {
        "settings": asdict(settings),
        "ingestion": ingestion,
        "integrity": integrity,
        "metrics": metrics,
    }
    result["statistics"] = statistics(
        settings.database, output, settings.bootstrap_replicates, settings.seed
    )
    if with_sensitivity:
        result["sensitivity"] = sensitivity(
            Path(settings.bundle) / "compact", settings.data, output
        )
        figures(settings.bundle, output, output / "figures")
    (output / "pipeline.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/platform.yaml")
    parser.add_argument("--with-sensitivity", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        result = run(Settings.load(args.config), args.with_sensitivity)
        print(
            json.dumps(
                {
                    "ingestion": result["ingestion"],
                    "integrity": result["integrity"],
                    "output": result["settings"]["output"],
                },
                indent=2,
            )
        )
    except (ValueError, OSError, sqlite3.Error) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
