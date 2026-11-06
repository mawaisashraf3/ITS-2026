import json
from pathlib import Path
import shutil
import sqlite3
import numpy as np
import pytest
from rmldnet.storage import ExperimentStore
from rmldnet.storage.validation import confusion, read_bundle
from rmldnet.analysis import cluster_bootstrap, f1_counts
from rmldnet.settings import Settings
from rmldnet.pipeline import run

BUNDLE = Path(__file__).resolve().parents[1] / "results/via"


def test_actual_bundle_ingestion_idempotence_and_metrics(tmp_path):
    with ExperimentStore(tmp_path / "db.sqlite3") as store:
        first = store.ingest(BUNDLE)
        second = store.ingest(BUNDLE)
        assert first["new_experiments"] == 4 and second["new_experiments"] == 0
        assert first["samples"] == 930 and first["observations"] == 7320
        assert store.integrity()["ok"]
        row = next(r for r in store.metrics("swin_b") if r["beta"] == 0)
        assert row["f1"] == pytest.approx(0.823403842168984)
        assert row["miou"] == pytest.approx(0.8390562403060476)
        assert not store.metrics("' OR 1=1 --")


def test_read_only_query_and_integrity_constraints(tmp_path):
    with ExperimentStore(tmp_path / "db.sqlite3") as store:
        store.ingest(BUNDLE)
        with pytest.raises(sqlite3.DatabaseError):
            store.query("DELETE FROM experiments RETURNING id")
        with pytest.raises(sqlite3.DatabaseError):
            store.query(f"ATTACH DATABASE '{tmp_path/'unwanted.sqlite3'}' AS other")
        assert not (tmp_path / "unwanted.sqlite3").exists()
        condition = store.connection.execute(
            "SELECT id FROM conditions LIMIT 1"
        ).fetchone()[0]
        sample = store.connection.execute(
            "SELECT m.sample_id FROM split_memberships m JOIN splits s ON s.id=m.split_id WHERE s.name='train' LIMIT 1"
        ).fetchone()[0]
        with pytest.raises(sqlite3.IntegrityError):
            store.connection.execute(
                "INSERT INTO observations VALUES(?,?,?,?,?,?)",
                (condition, sample, 0, 0, 0, 1),
            )
        store.connection.rollback()
        with pytest.raises(sqlite3.IntegrityError):
            store.connection.execute(
                "UPDATE conditions SET tp=-1 WHERE id=?", (condition,)
            )
        store.connection.rollback()
        with pytest.raises(sqlite3.IntegrityError):
            store.connection.execute(
                "UPDATE conditions SET tp=0.5 WHERE id=?", (condition,)
            )
        store.connection.rollback()
        assert store.integrity()["ok"]


def test_backup_recovery_and_export_roundtrip(tmp_path):
    original = tmp_path / "original.sqlite3"
    backup = tmp_path / "backup.sqlite3"
    with ExperimentStore(original) as store:
        store.ingest(BUNDLE)
        counts = store.counts()
        metrics = store.metrics()
        store.backup(backup)
        store.export_json(tmp_path / "export.json")
        with pytest.raises(FileExistsError):
            store.backup(backup)
    assert ExperimentStore.recover(backup, tmp_path / "recovered.sqlite3") == counts
    with ExperimentStore(tmp_path / "recovered.sqlite3") as recovered:
        assert recovered.metrics() == metrics
    with ExperimentStore(tmp_path / "imported.sqlite3") as store:
        assert store.import_json(tmp_path / "export.json") == counts
        assert store.metrics() == metrics
        with pytest.raises(ValueError, match="empty"):
            store.import_json(tmp_path / "export.json")


def test_invalid_bundle_does_not_mutate_store(tmp_path):
    bundle = tmp_path / "bundle"
    shutil.copytree(BUNDLE, bundle)
    path = bundle / "compact/results.json"
    data = json.loads(path.read_text())
    data["conditions"][0]["confusion"][0][0] += 1
    path.write_text(json.dumps(data))
    with ExperimentStore(tmp_path / "db.sqlite3") as store:
        with pytest.raises(ValueError, match="Aggregate"):
            store.ingest(bundle)
        assert all(n == 0 for n in store.counts().values())


def test_import_failure_rolls_back_every_table(tmp_path):
    with ExperimentStore(tmp_path / "source.sqlite3") as source:
        source.ingest(BUNDLE)
        source.export_json(tmp_path / "export.json")
    data = json.loads((tmp_path / "export.json").read_text())
    data["tables"]["conditions"][0]["tp"] = -1
    (tmp_path / "bad.json").write_text(json.dumps(data))
    with ExperimentStore(tmp_path / "target.sqlite3") as target:
        with pytest.raises(sqlite3.IntegrityError):
            target.import_json(tmp_path / "bad.json")
        assert all(v == 0 for v in target.counts().values())


def test_split_overlap_rejected(tmp_path):
    bundle = tmp_path / "bundle"
    shutil.copytree(BUNDLE, bundle)
    train = (bundle / "train_manifest.jsonl").read_text().splitlines()[0]
    path = bundle / "test_manifest.jsonl"
    path.write_text(train + "\n" + path.read_text())
    with pytest.raises(ValueError, match="overlap"):
        read_bundle(bundle)


def test_confusion_validation_and_bootstrap():
    with pytest.raises(ValueError):
        confusion([[1.2, 0], [0, 1]])
    with pytest.raises(ValueError):
        confusion([[-1, 0], [0, 1]])
    counts = np.array([[10, 2, 1, 5], [8, 1, 2, 4], [20, 0, 1, 7]])
    a, scores = cluster_bootstrap(counts, 200, 7)
    b, again = cluster_bootstrap(counts, 200, 7)
    assert a == b
    np.testing.assert_array_equal(scores, again)
    assert a["low"] <= a["point"] <= a["high"]
    np.testing.assert_allclose(f1_counts(counts), [10 / 13, 8 / 11, 14 / 15])
    with pytest.raises(ValueError):
        cluster_bootstrap(counts[:1])


def test_pipeline_database_analysis_integration(tmp_path):
    settings = Settings(
        database=str(tmp_path / "platform.sqlite3"),
        bundle=str(BUNDLE),
        output=str(tmp_path / "report"),
        bootstrap_replicates=200,
    )
    result = run(settings)
    assert result["integrity"]["ok"] and len(result["statistics"]["conditions"]) == 12
    assert (tmp_path / "report/statistics.csv").exists()
    repeated = run(settings)
    assert repeated["ingestion"]["new_experiments"] == 0


def test_settings_validation(tmp_path, monkeypatch):
    monkeypatch.delenv("RMLDNET_DB", raising=False)
    p = tmp_path / "settings.yaml"
    p.write_text("unknown: true\n")
    with pytest.raises(ValueError, match="Unknown"):
        Settings.load(p)
    p.write_text("seed: -1\n")
    with pytest.raises(ValueError):
        Settings.load(p)


def test_observation_updates_are_guarded(tmp_path):
    with ExperimentStore(tmp_path / "db.sqlite3") as store:
        store.ingest(BUNDLE)
        sample = store.connection.execute(
            "SELECT m.sample_id FROM split_memberships m JOIN splits s ON s.id=m.split_id WHERE s.name='train' LIMIT 1"
        ).fetchone()[0]
        condition, old_sample = store.connection.execute(
            "SELECT condition_id,sample_id FROM observations LIMIT 1"
        ).fetchone()
        with pytest.raises(sqlite3.IntegrityError):
            store.connection.execute(
                "UPDATE observations SET sample_id=? WHERE condition_id=? AND sample_id=?",
                (sample, condition, old_sample),
            )
        store.connection.rollback()
        assert store.integrity()["ok"]


def test_actual_sensitivity_matches_reference_confusions():
    root = BUNDLE.parent
    sweep = json.loads((root / "analysis/sensitivity.json").read_text())
    reference = json.loads((BUNDLE / "compact/results.json").read_text())
    threshold = sweep["fixed_validation_threshold"]
    for beta in [0, 1.2]:
        row = next(
            r
            for r in sweep["rows"]
            if r["beta"] == beta and r["threshold"] == threshold
        )
        expected = next(r for r in reference["conditions"] if r["beta"] == beta)
        assert [[row["tn"], row["fp"]], [row["fn"], row["tp"]]] == expected["confusion"]
        assert row["f1"] == expected["f1"]
