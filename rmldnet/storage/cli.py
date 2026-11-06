"""JSON-output database administration commands."""

import argparse
import json
import logging
import os
import sqlite3
from .store import ExperimentStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", default=os.environ.get("RMLDNET_DB", "runs/rmldnet.sqlite3")
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO"
    )
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ["init", "counts", "integrity"]:
        subs.add_parser(name)
    ingest = subs.add_parser("ingest")
    ingest.add_argument("bundle")
    metrics = subs.add_parser("metrics")
    metrics.add_argument("--model")
    query = subs.add_parser("query")
    query.add_argument("sql")
    for name in ["export", "import", "backup", "recover"]:
        command = subs.add_parser(name)
        command.add_argument("path")
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        if args.command == "recover":
            result = ExperimentStore.recover(args.path, args.database)
        else:
            with ExperimentStore(args.database) as store:
                if args.command in ("init", "counts"):
                    result = store.counts()
                elif args.command == "integrity":
                    result = store.integrity()
                    if not result["ok"]:
                        raise ValueError("Registry integrity check failed")
                elif args.command == "ingest":
                    result = store.ingest(args.bundle)
                elif args.command == "metrics":
                    result = store.metrics(args.model)
                elif args.command == "query":
                    result = store.query(args.sql)
                elif args.command == "export":
                    result = store.export_json(args.path)
                elif args.command == "import":
                    result = store.import_json(args.path)
                else:
                    result = store.backup(args.path)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, sqlite3.Error) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
