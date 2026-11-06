import json
from pathlib import Path
import pytest
from rmldnet.temporal import load_sequence, render_sequence, run_sequence


def test_temporal_order_requires_explicit_source_metadata(tmp_path):
    p = tmp_path / "sequence.json"
    p.write_text(json.dumps({"frames": []}))
    with pytest.raises(ValueError, match="source_kind"):
        load_sequence(p)
    p = render_sequence(tmp_path / "render", count=3)
    doc = load_sequence(p)
    doc["frames"][1]["timestamp_s"] = 0
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="timestamps"):
        load_sequence(p)


def test_controlled_images_execute_trained_segmentation_geometry_and_filter(tmp_path):
    p = render_sequence(tmp_path / "render", count=6)
    experiment = Path(__file__).resolve().parents[1] / "results/via/compact"
    report = run_sequence(p, experiment, tmp_path / "output")
    assert report["source_kind"] == "synthetic_controlled" and len(report["rows"]) == 6
    assert all("geometry" in row and "filter" in row for row in report["rows"])
    assert sum(row["filter"]["state"] is not None for row in report["rows"]) > 0
    assert (tmp_path / "output/sequence.json").exists()
