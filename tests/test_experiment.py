import numpy as np
import torch
from torch.utils.data import DataLoader
from rmldnet.baselines import UNet
from rmldnet.experiment import choose_threshold, measure, perturb
from rmldnet.losses import objective


def test_semantic_labels_do_not_supervise_instances():
    logits = torch.randn(2, 1, 8, 16, requires_grad=True)
    embeddings = torch.randn(2, 4, 8, 16, requires_grad=True)
    labels = torch.ones(2, 8, 16, dtype=torch.long)
    loss, parts = objective(
        {"logits": logits, "embeddings": embeddings},
        labels,
        torch.tensor([False, False]),
    )
    assert parts["instance"] == 0
    loss.backward()
    assert embeddings.grad.abs().sum() == 0


def test_unet_gradient_and_fog_equation():
    torch.set_num_threads(2)
    model = UNet()
    x = torch.rand(2, 3, 33, 65)
    result = model(x)
    assert result["logits"].shape == (2, 1, 33, 65)
    loss, _ = objective(
        result, torch.zeros(2, 33, 65, dtype=torch.long), torch.tensor([False, False])
    )
    loss.backward()
    assert model.out.weight.grad.abs().sum() > 0
    assert torch.equal(perturb(x, 0), x)
    fog = perturb(x, 1.2)
    assert fog.shape == x.shape and torch.isfinite(fog).all()


def test_threshold_selection_from_validation_predictions():
    p = np.array([[[0.2, 0.3, 0.7, 0.8]]])
    t = np.array([[[False, False, True, True]]])
    assert 0.3 < choose_threshold([(p, t)]) < 0.7


def test_measured_confusion_samples():
    class Fixed(torch.nn.Module):
        def forward(self, x):
            return {
                "logits": torch.tensor([[[[-8.0, 8.0], [-8.0, 8.0]]]]).expand(
                    len(x), -1, -1, -1
                )
            }

    samples = [
        {"image": torch.zeros(3, 2, 2), "mask": torch.tensor([[0, 1], [0, 1]])}
        for _ in range(3)
    ]
    metrics, cache = measure(Fixed(), DataLoader(samples, batch_size=2), collect=True)
    assert (
        metrics["f1"] == 1
        and metrics["samples"] == 3
        and metrics["confusion"] == [[6, 0], [0, 6]]
    )
    assert len(cache) == 2


def test_public_archive_checksum_masks_and_dedup(tmp_path, monkeypatch):
    from io import BytesIO
    from zipfile import ZipFile
    from PIL import Image
    from rmldnet import public_data
    import json

    archive = tmp_path / "fixture.zip"

    def png(color):
        buf = BytesIO()
        Image.new("RGB", (8, 8), color).save(buf, format="PNG")
        return buf.getvalue()

    with ZipFile(archive, "w") as z:
        for split, index, color in [
            ("train", 1, (10, 20, 30)),
            ("train", 101, (20, 30, 40)),
            ("val", 1, (30, 40, 50)),
            ("val", 2, (10, 20, 30)),
        ]:
            stem = f"{split}_{index:05d}.png"
            prefix = "Segmentation_Dataset/GGDataSet/"
            z.writestr(f"{prefix}{split}_frames/{split}/{stem}", png(color))
            z.writestr(f"{prefix}{split}_masks/{split}/{stem}", png((192, 128, 128)))
    monkeypatch.setattr(public_data, "SHA256", public_data.digest(archive))
    p = public_data.prepare(archive, tmp_path / "data", 1, 1)
    assert p["used"] == {"train": 1, "validation": 1, "test": 1}
    assert len(p["duplicates_removed"]) == 1
    row = json.loads((tmp_path / "data/train.jsonl").read_text())
    assert not row["instance_supervision"]
    assert np.asarray(Image.open(tmp_path / "data" / row["mask"])).min() == 1
    monkeypatch.setattr(public_data, "SHA256", "0" * 64)
    import pytest

    with pytest.raises(ValueError, match="SHA256"):
        public_data.prepare(archive, tmp_path / "bad")


def test_published_checkpoint_inference(tmp_path):
    import json
    from PIL import Image
    from rmldnet.predict import predict

    model = UNet()
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    (experiment / "config.json").write_text(
        json.dumps({"model": "unet", "size": [32, 64]})
    )
    torch.save({"state": model.state_dict(), "threshold": 0.5}, experiment / "best.pt")
    image = tmp_path / "image.png"
    Image.new("RGB", (100, 50), (50, 60, 70)).save(image)
    rows = predict(experiment, [image], tmp_path / "out")
    assert len(rows) == 1
    assert Image.open(tmp_path / "out/000000_mask.png").size == (100, 50)
    assert Image.open(tmp_path / "out/000000_overlay.png").size == (100, 50)


def test_raw_per_image_export_preserves_sample_ids(tmp_path):
    import json
    from rmldnet.experiment import write_results

    summary = {"conditions": [{"beta": 0, "per_image": [{"f1": 1.0}, {"f1": 0.5}]}]}
    write_results(tmp_path, summary, ["frame-a", "frame-b"])
    aggregate = json.loads((tmp_path / "results.json").read_text())
    rows = [
        json.loads(line)
        for line in (tmp_path / "per_image_metrics.jsonl").read_text().splitlines()
    ]
    assert [r["id"] for r in rows] == ["frame-a", "frame-b"]
    assert rows[1]["metrics"]["f1"] == 0.5
    assert "per_image" not in aggregate["conditions"][0]
