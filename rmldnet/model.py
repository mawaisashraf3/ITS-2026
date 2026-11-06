"""Atmospheric restoration, hierarchical shifted-window encoder and FPN."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.swin_transformer import SwinTransformer


class Defogger(nn.Module):
    """Assumed lightweight estimator; train with paired synthetic fog/clear images."""

    def __init__(self, width=16, t_min=0.1):
        super().__init__()
        self.t_min = t_min
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
        )
        self.transmission = nn.Conv2d(width, 1, 1)
        self.airlight = nn.Linear(width, 3)

    def forward(self, image):
        f = self.features(image)
        t = self.t_min + (1 - self.t_min) * self.transmission(f).sigmoid()
        a = self.airlight(f.mean((2, 3))).sigmoid()[:, :, None, None]
        restored = ((image - a) / t + a).clamp(0, 1)
        return restored, t, a


class Separable(nn.Sequential):
    def __init__(self, channels):
        super().__init__(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
        )


class RMLDNet(nn.Module):
    def __init__(
        self, variant="swin_b", fpn_channels=128, embedding_dim=4, defog=True, fpn=True
    ):
        super().__init__()
        if variant not in {"swin_b", "smoke"}:
            raise ValueError("variant must be swin_b or smoke")
        dim, depths, heads = (
            (128, [2, 2, 18, 2], [4, 8, 16, 32])
            if variant == "swin_b"
            else (16, [2, 2, 2, 2], [1, 2, 4, 8])
        )
        backbone = SwinTransformer(
            patch_size=[4, 4],
            embed_dim=dim,
            depths=depths,
            num_heads=heads,
            window_size=[7, 7],
            stochastic_depth_prob=0.2,
        )
        self.encoder = backbone.features
        self.defogger = Defogger() if defog else None
        self.use_fpn = fpn
        self.laterals = nn.ModuleList(
            nn.Conv2d(dim * 2**i, fpn_channels, 1) for i in range(4)
        )
        self.smooth = nn.ModuleList(Separable(fpn_channels) for _ in range(4))
        self.head = Separable(fpn_channels)
        self.classifier = nn.Conv2d(fpn_channels, 1, 1)
        self.embedding = nn.Conv2d(fpn_channels, embedding_dim, 1)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        )

    def load_encoder(self, path):
        """Load local torchvision-format weights; never silently ignore encoder keys."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        state = {
            k.removeprefix("features."): v
            for k, v in state.items()
            if k.startswith("features.")
        }
        if not state:
            raise ValueError(
                "Expected torchvision features.* keys; convert other formats explicitly"
            )
        self.encoder.load_state_dict(state, strict=True)

    def forward(self, image):
        size = image.shape[-2:]
        restored = self.defogger(image)[0] if self.defogger is not None else image
        x = (restored - self.mean) / self.std
        features = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in (1, 3, 5, 7):
                features.append(x.permute(0, 3, 1, 2).contiguous())
        p = [layer(f) for layer, f in zip(self.laterals, features)]
        if self.use_fpn:
            for i in (2, 1, 0):
                p[i] = p[i] + F.interpolate(
                    p[i + 1], size=p[i].shape[-2:], mode="bilinear", align_corners=False
                )
            fused = (
                sum(
                    F.interpolate(
                        s(f), size=p[0].shape[-2:], mode="bilinear", align_corners=False
                    )
                    for s, f in zip(self.smooth, p)
                )
                / 4
            )
        else:
            fused = F.interpolate(
                self.smooth[-1](p[-1]),
                size=p[0].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        h = self.head(fused)
        return {
            "logits": F.interpolate(
                self.classifier(h), size=size, mode="bilinear", align_corners=False
            ),
            "embeddings": F.interpolate(
                self.embedding(h), size=size, mode="bilinear", align_corners=False
            ),
            "restored": restored,
        }
