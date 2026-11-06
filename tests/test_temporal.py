import numpy as np
from rmldnet.tracking import LaneTracker


def test_stationary_lane_filter_reduces_measurement_jitter():
    rng = np.random.default_rng(7)
    truth = np.array([0.4, 0.02, 0.01, 0.0])
    measurements = truth + rng.normal(0, 0.02, (100, 4))
    measurements[:, 3] = 0
    tracker = LaneTracker(process_noise=1e-5, measurement_noise=0.0004)
    filtered = []
    for m in measurements:
        tracks = tracker.update([m])
        assert len(tracks) == 1
        filtered.append(tracks[0]["state"])
        p = tracker.tracks[0]["cov"]
        np.testing.assert_allclose(p, p.T, atol=1e-12)
        assert np.linalg.eigvalsh(p).min() > 0
    assert np.mean((np.array(filtered)[10:, 0] - truth[0]) ** 2) < np.mean(
        (measurements[10:, 0] - truth[0]) ** 2
    )


def test_gate_rejects_far_lane_and_reset():
    tracker = LaneTracker(gate=0.1)
    assert tracker.update([[0.1, 0, 0, 0]])[0]["id"] == 1
    tracks = tracker.update([[0.9, 0, 0, 0]])
    assert len(tracks) == 2 and tracks[0]["predicted"]
    tracker.reset()
    assert tracker.update([]) == []
