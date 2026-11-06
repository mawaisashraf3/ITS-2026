"""Small U-Net reference baseline trained under the same pixel protocol."""

from torch import nn
from torch.nn import functional as F
import torch


def block(a, b):
    return nn.Sequential(
        nn.Conv2d(a, b, 3, padding=1),
        nn.GELU(),
        nn.Conv2d(b, b, 3, padding=1),
        nn.GELU(),
    )


class UNet(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.e1 = block(3, width)
        self.e2 = block(width, width * 2)
        self.e3 = block(width * 2, width * 4)
        self.bottom = block(width * 4, width * 8)
        self.d3 = block(width * 12, width * 4)
        self.d2 = block(width * 6, width * 2)
        self.d1 = block(width * 3, width)
        self.out = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        b = self.bottom(F.max_pool2d(e3, 2))

        def up(a, b):
            return torch.cat(
                [
                    F.interpolate(
                        a, size=b.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    b,
                ],
                1,
            )

        d = self.d1(up(self.d2(up(self.d3(up(b, e3)), e2)), e1))
        logits = self.out(d)
        return {"logits": logits, "embeddings": logits * 0, "restored": x}
