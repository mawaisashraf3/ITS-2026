"""Equations 7–9 plus explicitly chosen discriminative embedding objective."""

import torch
from torch.nn import functional as F


def instance_loss(embeddings, labels, delta_v=0.5, delta_d=1.5):
    total = embeddings.sum() * 0
    for emb, lab in zip(embeddings, labels):
        clusters = [emb[:, lab == i].T for i in torch.unique(lab) if i > 0]
        if not clusters:
            continue
        means = torch.stack([c.mean(0) for c in clusters])
        variance = torch.stack(
            [
                (c - m).norm(dim=1).sub(delta_v).clamp_min(0).square().mean()
                for c, m in zip(clusters, means)
            ]
        ).mean()
        distance = means.sum() * 0
        if len(clusters) > 1:
            d = torch.cdist(means, means)
            mask = ~torch.eye(len(clusters), device=d.device, dtype=torch.bool)
            distance = (2 * delta_d - d[mask]).clamp_min(0).square().mean()
        total = total + variance + distance + 0.001 * means.norm(dim=1).mean()
    return total / embeddings.shape[0]


def objective(output, labels, instance_valid=None):
    target = (labels > 0).float()[:, None]
    p = output["logits"].sigmoid()
    dims = (1, 2, 3)
    intersection = (p * target).sum(dims)
    dice = (
        1 - ((2 * intersection + 1e-6) / (p.sum(dims) + target.sum(dims) + 1e-6)).mean()
    )
    iou = (
        1
        - (
            (intersection + 1e-6)
            / (p.sum(dims) + target.sum(dims) - intersection + 1e-6)
        ).mean()
    )
    bce = F.binary_cross_entropy_with_logits(output["logits"], target)
    if instance_valid is None:
        inst = instance_loss(output["embeddings"], labels)
    elif instance_valid.any():
        inst = instance_loss(
            output["embeddings"][instance_valid], labels[instance_valid]
        )
    else:
        inst = output["embeddings"].sum() * 0
    return bce + dice + 0.5 * inst + 0.5 * iou, {
        "bce": bce,
        "dice": dice,
        "instance": inst,
        "iou_loss": iou,
    }
