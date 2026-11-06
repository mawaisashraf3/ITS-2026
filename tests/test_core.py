import json
import numpy as np
import pytest
import torch
from PIL import Image
from rmldnet.model import RMLDNet, Defogger
from rmldnet.losses import objective, instance_loss
from rmldnet.metrics import Confusion
from rmldnet.tracking import LaneTracker, lane_state, state_points, decode_lanes
from rmldnet.data import synthetic, LaneDataset
from rmldnet.prepare import convert, bdd_vertices


def test_full_gradient_and_odd_shape():
    torch.set_num_threads(2)
    model = RMLDNet("smoke", 16)
    x = torch.rand(2, 3, 65, 129)
    labels = torch.zeros(2, 65, 129, dtype=torch.long)
    labels[:, :, 30:33] = 1
    labels[:, :, 80:83] = 2
    result = model(x)
    loss, parts = objective(result, labels)
    assert result["logits"].shape == (2, 1, 65, 129)
    assert torch.isfinite(loss)
    loss.backward()
    for module in [model.defogger, model.encoder, model.classifier, model.embedding]:
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        )


def test_restoration_equation():
    model = Defogger()
    image = torch.rand(1, 3, 16, 32)
    restored, t, a = model(image)
    assert t.min() >= 0.1 and t.max() <= 1
    torch.testing.assert_close(restored, ((image - a) / t + a).clamp(0, 1))


def test_metrics_known_confusion():
    metric = Confusion()
    metric.update([1, 1, 0, 0], [1, 0, 1, 0])
    m = metric.compute()
    assert m["f1"] == 0.5 and m["lane_iou"] == pytest.approx(1 / 3)
    assert m["miou"] == pytest.approx(1 / 3) and m["accuracy"] == 0.5
    empty = Confusion()
    empty.update([0], [0])
    assert empty.compute()["f1"] is None


def test_embedding_permutation_and_empty():
    emb = torch.randn(1, 4, 8, 8, requires_grad=True)
    lab = torch.zeros(1, 8, 8, dtype=torch.long)
    loss = instance_loss(emb, lab)
    loss.backward()
    assert loss == 0
    lab[:, :, :3] = 1
    lab[:, :, 5:] = 2
    swapped = lab.clone()
    swapped[lab == 1] = 2
    swapped[lab == 2] = 1
    torch.testing.assert_close(instance_loss(emb, lab), instance_loss(emb, swapped))


def test_tracker_association_expiry_and_covariance():
    tracker = LaneTracker(max_age=2)
    initial = tracker.update([[0.2, 0, 0, 0], [0.8, 0, 0, 0]])
    updated = tracker.update([[0.79, 0, 0, 0], [0.21, 0, 0, 0]])
    assert [x["id"] for x in initial] == [x["id"] for x in updated]
    assert updated[0]["state"][0] < 0.3
    for _ in range(2):
        assert len(tracker.update([])) == 2
    assert tracker.update([]) == []
    assert all(np.linalg.eigvalsh(t["cov"]).min() >= 0 for t in tracker.tracks)


def test_geometry_and_decode():
    s = np.array([0.3, 0.1, 0.02, 0.001])
    ys = np.linspace(0.3, 1, 30)
    np.testing.assert_allclose(lane_state(state_points(s, ys)), s, atol=1e-10)
    prob = np.zeros((20, 30))
    prob[:, 5] = 1
    prob[:, 24] = 1
    emb = np.zeros((2, 20, 30))
    emb[:, :, 24] = 4
    assert len(decode_lanes(prob, emb)) == 2


def test_synthetic_and_tusimple_conversion(tmp_path):
    synthetic(tmp_path / "toy", 8)
    ds = LaneDataset(tmp_path / "toy/train.jsonl", (64, 128), True)
    assert ds[0]["mask"].max() == 3 and len(ds) == 6
    Image.new("RGB", (32, 20)).save(tmp_path / "frame.png")
    ann = tmp_path / "ann.jsonl"
    ann.write_text(
        json.dumps(
            {
                "raw_file": "frame.png",
                "lanes": [[5, 6, 7, -2]],
                "h_samples": [1, 5, 10, 15],
            }
        )
        + "\n"
    )
    convert("tusimple", ann, tmp_path, tmp_path / "converted")
    converted = LaneDataset(tmp_path / "converted/manifest.jsonl", (20, 32))
    assert converted[0]["mask"].max() == 1


def test_culane_and_bdd_conversion(tmp_path):
    Image.new("RGB", (32, 20)).save(tmp_path / "frame.jpg")
    (tmp_path / "frame.lines.txt").write_text("2 1 3 5 4 10\n")
    (tmp_path / "list.txt").write_text("/frame.jpg 0 0 1\n")
    convert("culane", tmp_path / "list.txt", tmp_path, tmp_path / "culane")
    poly = {"vertices": [[1, 1], [2, 3], [4, 6], [8, 10]], "types": "LCCL"}
    assert len(bdd_vertices(poly)) == 20
    ann = [
        {
            "name": "frame.jpg",
            "labels": [{"category": "lane/single white", "poly2d": [poly]}],
        }
    ]
    (tmp_path / "bdd.json").write_text(json.dumps(ann))
    convert("bdd100k", tmp_path / "bdd.json", tmp_path, tmp_path / "bdd")
    assert LaneDataset(tmp_path / "bdd/manifest.jsonl", (20, 32))[0]["mask"].max() == 1


def test_ablations_and_checkpoint(tmp_path):
    model = RMLDNet("smoke", 16, defog=False, fpn=False).eval()
    with torch.inference_mode():
        out = model(torch.rand(1, 3, 32, 64))
    assert out["logits"].shape == (1, 1, 32, 64)
    path = tmp_path / "encoder.pt"
    torch.save(
        {"features." + k: v for k, v in model.encoder.state_dict().items()}, path
    )
    model.load_encoder(path)
