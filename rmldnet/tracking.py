"""Image-coordinate proxy state and gated Hungarian association (assumptions)."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from .kalman import LinearKalmanFilter


def lane_state(points):
    """Fit x(y) in normalized coordinates; bottom reference y=1. No metric calibration."""
    pts = np.asarray(points, float)
    if len(pts) < 4:
        raise ValueError("At least four points required")
    q = pts[:, 1] - 1
    a, b, c, d = np.polyfit(q, pts[:, 0], 3)
    return np.array([d, np.arctan(c), 2 * b, 6 * a])


def state_points(state, ys):
    d, heading, curvature, rate = state
    q = np.asarray(ys) - 1
    return np.stack(
        [d + np.tan(heading) * q + curvature * q * q / 2 + rate * q**3 / 6, ys], axis=1
    )


class LaneTracker:
    def __init__(
        self, gate=0.25, max_age=3, process_noise=1e-4, measurement_noise=1e-2
    ):
        self.gate, self.max_age = gate, max_age
        self.q = np.eye(4) * process_noise
        self.r = np.eye(4) * measurement_noise
        self.tracks = []
        self.next_id = 1

    def reset(self):
        self.tracks = []
        self.next_id = 1

    def update(self, measurements, dt=1.0):
        z = np.asarray(measurements, float).reshape(-1, 4)
        # Near-constant curvature: k(t+dt)=k(t)+dt*k_dot; other states random walks.
        f = np.eye(4)
        f[2, 3] = dt
        for t in self.tracks:
            t["_filter"] = LinearKalmanFilter(
                t["state"], t["cov"], f, np.eye(4), self.q, self.r
            )
            t["state"] = t["_filter"].predict()
            t["cov"] = t["_filter"].P.copy()
            t["age"] += 1
        matched = set()
        if self.tracks and len(z):
            cost = np.array(
                [
                    [
                        abs(t["state"][0] - m[0]) + 0.1 * abs(t["state"][1] - m[1])
                        for m in z
                    ]
                    for t in self.tracks
                ]
            )
            # Dummy columns permit unmatched assignments without stealing valid matches.
            gated = np.where(cost <= self.gate, cost, 1e6)
            augmented = np.concatenate(
                [
                    gated,
                    np.full((len(self.tracks), len(self.tracks)), self.gate + 1e-6),
                ],
                axis=1,
            )
            rows, cols = linear_sum_assignment(augmented)
            for i, j in zip(rows, cols):
                if j >= len(z) or cost[i, j] > self.gate:
                    continue
                t = self.tracks[i]
                t["_filter"].update(z[j])
                t["state"] = t["_filter"].state
                t["cov"] = t["_filter"].P.copy()
                t["age"] = 0
                matched.add(j)
        self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]
        for j, m in enumerate(z):
            if j not in matched:
                self.tracks.append(
                    {
                        "id": self.next_id,
                        "state": m.copy(),
                        "cov": np.eye(4) * 0.1,
                        "age": 0,
                    }
                )
                self.next_id += 1
        return [
            {"id": t["id"], "state": t["state"].copy(), "predicted": t["age"] > 0}
            for t in self.tracks
        ]


def decode_lanes(
    probability, embeddings, threshold=0.5, bandwidth=1.0, min_pixels=12, max_lanes=8
):
    """Deterministic greedy embedding clustering; approximate, configured explicitly."""
    ys, xs = np.where(np.asarray(probability) > threshold)
    if not len(ys):
        return []
    values = embeddings[:, ys, xs].T
    centers = []
    groups = []
    for n, v in enumerate(values):
        distances = (
            np.linalg.norm(np.asarray(centers) - v, axis=1) if centers else np.array([])
        )
        if len(distances) and distances.min() < bandwidth:
            k = int(distances.argmin())
            groups[k].append(n)
            centers[k] += (v - centers[k]) / len(groups[k])
        elif len(centers) < max_lanes:
            centers.append(v.copy())
            groups.append([n])
    lanes = []
    h, w = probability.shape
    for group in groups:
        if len(group) < min_pixels:
            continue
        gy, gx = ys[group], xs[group]
        pts = np.array([(np.median(gx[gy == y]) / w, y / h) for y in np.unique(gy)])
        if len(pts) >= 4:
            lanes.append(pts)
    return lanes
