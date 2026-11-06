"""Conservative, instance-free corridor measurements from semantic lane masks."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GeometryConfig:
    near_y: float = 0.85
    far_y: float = 0.60
    band_radius: int = 2
    min_rows: int = 2
    max_mark_width: float = 0.15
    min_corridor_width: float = 0.12
    max_corridor_width: float = 0.98

    def __post_init__(self):
        if type(self.band_radius) is not int or type(self.min_rows) is not int:
            raise ValueError("Row-band radius and minimum rows must be integers")
        if not 0 < self.far_y < self.near_y < 1:
            raise ValueError("Need 0 < far_y < near_y < 1")
        if (
            self.band_radius < 0
            or self.min_rows < 1
            or self.min_rows > 2 * self.band_radius + 1
        ):
            raise ValueError("Invalid row-band configuration")
        if (
            not 0 < self.max_mark_width < 1
            or not 0 < self.min_corridor_width < self.max_corridor_width <= 1
        ):
            raise ValueError("Invalid geometry width limits")


def row_segments(row):
    """Return inclusive start/exclusive end of every foreground run."""
    row = np.asarray(row, dtype=bool)
    if row.ndim != 1:
        raise ValueError("Expected one mask row")
    padded = np.r_[False, row, False].astype(int)
    edges = np.diff(padded)
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def extract_corridor(probability, threshold=0.5, config=None):
    config = config or GeometryConfig()
    p = np.asarray(probability, dtype=float)
    if (
        p.ndim != 2
        or min(p.shape) < 8
        or not np.isfinite(p).all()
        or p.min() < 0
        or p.max() > 1
    ):
        raise ValueError("Expected a finite probability map in [0,1]")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold out of range")
    h, w = p.shape
    anchors = []
    for y in [config.near_y, config.far_y]:
        central = int(round(y * (h - 1)))
        measurements = []
        widths = []
        confidence = []
        for row in range(
            max(0, central - config.band_radius),
            min(h, central + config.band_radius + 1),
        ):
            runs = [
                (a, b)
                for a, b in row_segments(p[row] > threshold)
                if (b - a) / w <= config.max_mark_width
            ]
            centers = [((a + b - 1) / 2 / (w - 1), a, b) for a, b in runs]
            left = [c for c in centers if c[0] < 0.5]
            right = [c for c in centers if c[0] > 0.5]
            if not left or not right:
                continue
            l = max(left, key=lambda c: c[0])
            r = min(right, key=lambda c: c[0])
            width = r[0] - l[0]
            if not config.min_corridor_width <= width <= config.max_corridor_width:
                continue
            measurements.append((l[0] + r[0]) / 2)
            widths.append(width)
            confidence.append(
                float(np.mean(np.r_[p[row, l[1] : l[2]], p[row, r[1] : r[2]]]))
            )
        valid = len(measurements) >= config.min_rows
        anchors.append(
            {
                "y": y,
                "valid": valid,
                "center": float(np.median(measurements)) if valid else None,
                "width": float(np.median(widths)) if valid else None,
                "valid_rows": len(measurements),
                "confidence": float(np.mean(confidence)) if valid else None,
                "reason": "ok" if valid else "insufficient_bracketing_boundaries",
            }
        )
    return {
        "centers": [a["center"] for a in anchors],
        "anchors": anchors,
        "both_valid": all(a["valid"] for a in anchors),
        "units": "normalized image width; corridor proxy, not metric vehicle position",
    }
