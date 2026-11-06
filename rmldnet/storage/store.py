"""Atomic experiment ingestion, SQL retrieval, portable exports and safe backups."""

import json
import logging
from pathlib import Path
import sqlite3
from .validation import TABLES, file_hash, read_bundle

LOGGER = logging.getLogger(__name__)


class ExperimentStore:
    """Single-workstation ACID registry; image bytes remain in external datasets."""

    def __init__(self, path="runs/rmldnet.sqlite3"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1):
            self.connection.close()
            raise ValueError(f"Unsupported schema version {version}")
        if (
            version == 0
            and self.connection.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchone()[0]
        ):
            self.connection.close()
            raise ValueError("Existing database has no recognized registry schema")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript(
            Path(__file__).with_name("schema.sql").read_text()
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.connection.close()

    def ingest(self, root):
        """Validate the whole bundle, then commit every table together or roll back."""
        root, metadata, manifests, key, runs = read_bundle(root)
        added = 0
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO datasets(dataset_key,name,metadata_json) VALUES(?,?,?)",
                (key, metadata["dataset"], json.dumps(metadata, sort_keys=True)),
            )
            dataset_id = self.connection.execute(
                "SELECT id FROM datasets WHERE dataset_key=?", (key,)
            ).fetchone()[0]
            for split, rows in manifests.items():
                self.connection.execute(
                    "INSERT OR IGNORE INTO splits(dataset_id,name) VALUES(?,?)",
                    (dataset_id, split),
                )
                split_id = self.connection.execute(
                    "SELECT id FROM splits WHERE dataset_id=? AND name=?",
                    (dataset_id, split),
                ).fetchone()[0]
                for row in rows:
                    self.connection.execute(
                        "INSERT OR IGNORE INTO samples(dataset_id,external_id,image_uri,mask_uri,sha256,group_id,instance_supervision) VALUES(?,?,?,?,?,?,?)",
                        (
                            dataset_id,
                            row["id"],
                            row["image"],
                            row["mask"],
                            row["sha256"],
                            row["group"],
                            int(row.get("instance_supervision", True)),
                        ),
                    )
                    sample_id = self.connection.execute(
                        "SELECT id FROM samples WHERE dataset_id=? AND external_id=?",
                        (dataset_id, row["id"]),
                    ).fetchone()[0]
                    self.connection.execute(
                        "INSERT OR IGNORE INTO split_memberships VALUES(?,?,?)",
                        (dataset_id, sample_id, split_id),
                    )
            sample_ids = {
                r["external_id"]: r["id"]
                for r in self.connection.execute(
                    "SELECT id,external_id FROM samples WHERE dataset_id=?",
                    (dataset_id,),
                )
            }
            for run in runs:
                if self.connection.execute(
                    "SELECT id FROM experiments WHERE run_key=?", (run["run_key"],)
                ).fetchone():
                    continue
                report = run["result"]
                directory = run["directory"]
                cur = self.connection.execute(
                    "INSERT INTO experiments(dataset_id,run_key,model,seed,state_sha256,config_json,result_sha256) VALUES(?,?,?,?,?,?,?)",
                    (
                        dataset_id,
                        run["run_key"],
                        report["model"],
                        report["seed"],
                        report["checkpoint_tensor_sha256"],
                        json.dumps(run["config"], sort_keys=True),
                        file_hash(directory / "results.json"),
                    ),
                )
                experiment_id = cur.lastrowid
                for condition in report["conditions"]:
                    tn, fp, fn, tp = [
                        int(v) for row in condition["confusion"] for v in row
                    ]
                    cur = self.connection.execute(
                        "INSERT INTO conditions(experiment_id,beta,threshold,sample_count,elapsed_seconds,tn,fp,fn,tp) VALUES(?,?,?,?,?,?,?,?,?)",
                        (
                            experiment_id,
                            condition["beta"],
                            condition["threshold"],
                            condition["samples"],
                            condition["elapsed_seconds"],
                            tn,
                            fp,
                            fn,
                            tp,
                        ),
                    )
                    condition_id = cur.lastrowid
                    self.connection.executemany(
                        "INSERT INTO observations VALUES(?,?,?,?,?,?)",
                        [
                            (condition_id, sample_ids[sample_id], *counts)
                            for sample_id, counts in run["observations"][
                                condition["beta"]
                            ].items()
                        ],
                    )
                for artifact in sorted(directory.iterdir()):
                    if artifact.is_file():
                        if not artifact.resolve().is_relative_to(root):
                            raise ValueError("Artifact symlink escapes bundle")
                        self.connection.execute(
                            "INSERT INTO artifacts(experiment_id,relative_path,kind,sha256,bytes) VALUES(?,?,?,?,?)",
                            (
                                experiment_id,
                                str(artifact.relative_to(root)),
                                artifact.suffix.lstrip("."),
                                file_hash(artifact),
                                artifact.stat().st_size,
                            ),
                        )
                added += 1
        LOGGER.info("Ingested %d new experiment runs", added)
        return {"new_experiments": added, **self.counts()}

    def counts(self):
        return {
            name: self.connection.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
            for name in TABLES
        }

    def metrics(self, model=None):
        sql = "SELECT * FROM metrics"
        params = ()
        if model is not None:
            sql += " WHERE model=?"
            params = (model,)
        return [
            dict(row)
            for row in self.connection.execute(
                sql + " ORDER BY model,seed,beta", params
            )
        ]

    def query(self, sql, parameters=()):
        """Execute user queries through a separate read-only connection."""
        with sqlite3.connect(
            self.path.resolve().as_uri() + "?mode=ro", uri=True
        ) as reader:
            reader.row_factory = sqlite3.Row
            allowed = {
                sqlite3.SQLITE_SELECT,
                sqlite3.SQLITE_READ,
                sqlite3.SQLITE_FUNCTION,
                sqlite3.SQLITE_RECURSIVE,
            }
            reader.set_authorizer(
                lambda action, *args: sqlite3.SQLITE_OK
                if action in allowed
                else sqlite3.SQLITE_DENY
            )
            cursor = reader.execute(sql, parameters)
            if cursor.description is None:
                raise ValueError("Query must return rows")
            return [dict(row) for row in cursor.fetchall()]

    def integrity(self):
        integrity = [r[0] for r in self.connection.execute("PRAGMA integrity_check")]
        foreign = [
            tuple(r) for r in self.connection.execute("PRAGMA foreign_key_check")
        ]
        mismatches = self.connection.execute("""SELECT c.id FROM conditions c LEFT JOIN observations o ON o.condition_id=c.id
          GROUP BY c.id HAVING count(o.sample_id)!=c.sample_count OR sum(o.tn)!=c.tn OR sum(o.fp)!=c.fp OR sum(o.fn)!=c.fn OR sum(o.tp)!=c.tp""").fetchall()
        invalid_associations = self.connection.execute("""SELECT o.condition_id,o.sample_id FROM observations o
          JOIN conditions c ON c.id=o.condition_id JOIN experiments e ON e.id=c.experiment_id
          JOIN samples s ON s.id=o.sample_id JOIN split_memberships m ON m.sample_id=s.id
          JOIN splits p ON p.id=m.split_id WHERE e.dataset_id!=s.dataset_id OR p.name!='test'""").fetchall()
        invalid_groups = self.connection.execute("""SELECT s.dataset_id,s.group_id FROM samples s
          JOIN split_memberships m ON m.sample_id=s.id GROUP BY s.dataset_id,s.group_id HAVING count(DISTINCT m.split_id)>1""").fetchall()
        return {
            "ok": integrity == ["ok"]
            and not foreign
            and not mismatches
            and not invalid_associations
            and not invalid_groups,
            "invalid_observation_associations": [list(r) for r in invalid_associations],
            "cross_split_groups": [list(r) for r in invalid_groups],
            "integrity_check": integrity,
            "foreign_key_violations": foreign,
            "aggregate_mismatches": [r[0] for r in mismatches],
        }

    def export_json(self, path):
        """Portable relational snapshot in dependency order; explicit destination."""
        self.connection.execute("BEGIN DEFERRED")
        try:
            payload = {
                "schema_version": 1,
                "tables": {
                    name: [
                        dict(r)
                        for r in self.connection.execute(
                            f"SELECT * FROM {name} ORDER BY 1"
                        )
                    ]
                    for name in TABLES
                },
            }
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(payload, stream, indent=2)
        return {"path": str(path), "sha256": file_hash(path), "tables": self.counts()}

    def import_json(self, path):
        """Restore a portable snapshot into an empty registry transactionally."""
        payload = json.loads(Path(path).read_text())
        if payload.get("schema_version") != 1 or set(payload.get("tables", {})) != set(
            TABLES
        ):
            raise ValueError("Unsupported or incomplete export")
        if any(self.counts().values()):
            raise ValueError("JSON import requires an empty database")
        with self.connection:
            for name in TABLES:
                columns = {
                    r[1] for r in self.connection.execute(f"PRAGMA table_info({name})")
                }
                for row in payload["tables"][name]:
                    if set(row) != columns:
                        raise ValueError(f"Unexpected columns in {name}")
                    names = sorted(row)
                    self.connection.execute(
                        f'INSERT INTO {name}({",".join(names)}) VALUES({",".join("?" for _ in names)})',
                        [row[n] for n in names],
                    )
            if not self.integrity()["ok"]:
                raise ValueError("Imported database failed integrity checks")
        return self.counts()

    def backup(self, path):
        """Use SQLite online backup so WAL state is copied consistently."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb"):
            pass
        try:
            with sqlite3.connect(path) as destination:
                self.connection.backup(destination)
            with sqlite3.connect(path) as destination:
                if destination.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ValueError("Backup integrity failure")
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return {
            "path": str(path),
            "sha256": file_hash(path),
            "bytes": path.stat().st_size,
        }

    @classmethod
    def recover(cls, backup, destination):
        backup = Path(backup)
        with sqlite3.connect(
            backup.resolve().as_uri() + "?mode=ro", uri=True
        ) as reader:
            if (
                reader.execute("PRAGMA user_version").fetchone()[0] != 1
                or reader.execute("PRAGMA integrity_check").fetchone()[0] != "ok"
            ):
                raise ValueError("Invalid registry backup")
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb"):
                pass
            try:
                with sqlite3.connect(destination) as target:
                    reader.backup(target)
                with cls(destination) as store:
                    if not store.integrity()["ok"]:
                        raise ValueError("Recovered registry failed integrity checks")
                    result = store.counts()
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
        return result
