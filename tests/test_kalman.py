import numpy as np
import pytest
from rmldnet.kalman import (
    LinearKalmanFilter,
    LaneFilterConfig,
    LanePositionFilter,
    lane_dynamics,
)
from rmldnet.geometry import extract_corridor, GeometryConfig, row_segments
from rmldnet.robustness import CalibrationAccumulator, geometry_error
from rmldnet.kalman_experiments import evaluate


def test_scalar_filter_matches_closed_form():
    kf = LinearKalmanFilter([0.0], [[1.0]], [[1.0]], [[1.0]], [[0.0]], [[1.0]])
    kf.predict()
    result = kf.update([1.0])
    assert result.accepted and result.normalized_innovation_squared == pytest.approx(
        0.5
    )
    np.testing.assert_allclose(result.gain, [[0.5]])
    np.testing.assert_allclose(kf.state, [0.5])
    np.testing.assert_allclose(kf.P, [[0.5]])


def test_control_and_rejected_observation_preserve_state():
    kf = LinearKalmanFilter(
        [0.0], [[1.0]], [[1.0]], [[1.0]], [[0.0]], [[1.0]], B=[[2.0]]
    )
    np.testing.assert_allclose(kf.predict(control=[3.0]), [6.0])
    prior = kf.state.copy()
    P = kf.P.copy()
    result = kf.update([100.0], gate=9.0)
    assert not result.accepted
    np.testing.assert_array_equal(kf.state, prior)
    np.testing.assert_array_equal(kf.P, P)
    with pytest.raises(ValueError):
        kf.update([np.nan])


def test_covariance_validation_and_dimensions():
    with pytest.raises(ValueError):
        LinearKalmanFilter([0.0], [[-1.0]], [[1.0]], [[1.0]], [[0.0]], [[1.0]])
    with pytest.raises(ValueError):
        LinearKalmanFilter([0.0], [[1.0]], [[1.0]], [[1.0]], [[0.0]], [[0.0]])
    kf = LinearKalmanFilter(
        [0.0, 0.0], np.eye(2), np.eye(2), np.eye(2), np.eye(2) * 0.1, np.eye(2)
    )
    with pytest.raises(ValueError):
        kf.update([1.0])
    with pytest.raises(ValueError):
        kf.predict(F=np.eye(3))
    with pytest.raises(ValueError):
        kf.predict(control=[1.0])


def test_lane_dynamics_matches_discrete_acceleration_equations():
    F, Q = lane_dynamics(0.2, 0.4)
    G = np.vstack([np.eye(2) * 0.02, np.eye(2) * 0.2])
    np.testing.assert_allclose(Q, 0.4 * G @ G.T)
    np.testing.assert_allclose(
        F @ np.array([0.4, 0.5, 0.1, -0.1]), [0.42, 0.48, 0.1, -0.1]
    )
    with pytest.raises(ValueError):
        lane_dynamics(0, 0.1)


def test_lane_partial_missing_reset_and_monotonic_timestamps():
    tracker = LanePositionFilter(LaneFilterConfig(initialization_observations=1))
    assert tracker.step([0.5, None], 0)["state"] is None
    initialized = tracker.step([0.5, 0.48], 0.1)
    prior = np.array(initialized["covariance"])
    predicted = tracker.step([None, None], 0.2)
    assert (
        predicted["status"] == "prediction_only"
        and np.array(predicted["covariance"])[0, 0] > prior[0, 0]
    )
    partial = tracker.step([0.51, None], 0.3)
    assert partial["status"] == "updated"
    with pytest.raises(ValueError):
        tracker.step([0.5, 0.5], 0.3)
    tracker.reset()
    assert tracker.step([None, None], 0)["state"] is None


def test_covariance_stays_symmetric_positive_under_many_updates():
    rng = np.random.default_rng(7)
    tracker = LanePositionFilter(LaneFilterConfig(initialization_observations=1))
    for i in range(500):
        state = tracker.step(
            (np.array([0.5, 0.48]) + rng.normal(0, 0.01, 2)).tolist(), i * 0.1
        )
        P = np.array(state["covariance"])
        np.testing.assert_allclose(P, P.T, atol=1e-12)
        assert np.linalg.eigvalsh(P).min() > -1e-12


def test_filter_config_rejects_invalid_noise():
    with pytest.raises(ValueError):
        LaneFilterConfig(measurement_variance=0)
    with pytest.raises(ValueError):
        LaneFilterConfig(gate=-1)


def test_corridor_measurements_and_missing_boundaries():
    p = np.zeros((100, 200))
    p[:, 38:41] = 1
    p[:, 158:161] = 1
    measurement = extract_corridor(p, 0.5)
    assert measurement["both_valid"]
    np.testing.assert_allclose(measurement["centers"], [99 / 199] * 2)
    p[:, 158:161] = 0
    assert not extract_corridor(p, 0.5)["both_valid"]
    assert not extract_corridor(np.ones((100, 200)), 0.5)["both_valid"]
    assert row_segments([0, 1, 1, 0, 1]) == [(1, 3), (4, 5)]


def test_geometry_validation_and_paired_error_units():
    with pytest.raises(ValueError):
        GeometryConfig(far_y=0.9)
    with pytest.raises(ValueError):
        extract_corridor(np.full((10, 10), np.nan))
    error = geometry_error({"centers": [0.51, None]}, {"centers": [0.5, 0.5]}, 201)
    assert (
        error["near_error_px"] == pytest.approx(2.0) and error["far_error_px"] is None
    )


def test_calibration_known_brier_nll_and_ece():
    cal = CalibrationAccumulator(10)
    cal.update(np.array([0.1, 0.8]), np.array([False, True]), 0.5)
    result = cal.compute()
    assert result["brier"] == pytest.approx(0.025)
    assert result["ece"] == pytest.approx(0.15)
    assert result["negative_log_likelihood"] == pytest.approx(-np.log(0.9 * 0.8) / 2)
    assert result["positive_brier"] == pytest.approx(0.04)
    with pytest.raises(ValueError):
        CalibrationAccumulator().compute()


def test_synthetic_filter_execution_is_deterministic_and_improves_seeded_stress():
    first, records = evaluate(7, True)
    second, _ = evaluate(7, True)
    assert first == second and len(records) == 240
    assert first["filtered_rmse_observed"] < first["raw_rmse_observed"]
    assert first["missing_frames"] > 0 and first["rejected_observations"] > 0


def test_reacquisition_recovers_preserved_initialization_failure():
    before, _ = evaluate(
        24, True, LaneFilterConfig(reacquire_after=None, initialization_observations=1)
    )
    intermediate, _ = evaluate(
        24, True, LaneFilterConfig(initialization_observations=1)
    )
    assert intermediate["reacquisitions"] == 1
    assert intermediate["filtered_rmse_observed"] < 0.04
    after, records = evaluate(24, True)
    assert before["filtered_rmse_observed"] > 1
    assert after["filtered_rmse_observed"] < 0.04
    assert after["initialized_scored_frames"] == after["frames_after_burn_in"]
    assert any(r["status"] == "waiting_for_stable_initialization" for r in records)


def test_correlated_state_update_matches_hand_calculation():
    kf = LinearKalmanFilter(
        [1.0, 2.0],
        [[2.0, 0.5], [0.5, 1.0]],
        np.eye(2),
        [[1.0, 1.0]],
        np.zeros((2, 2)),
        [[0.5]],
    )
    result = kf.update([4.0])
    np.testing.assert_allclose(result.gain, [[5 / 9], [1 / 3]])
    np.testing.assert_allclose(kf.state, [14 / 9, 7 / 3])
    np.testing.assert_allclose(kf.P, [[11 / 18, -1 / 3], [-1 / 3, 0.5]], atol=1e-12)


def test_stable_window_defers_a_bad_first_measurement():
    tracker = LanePositionFilter(LaneFilterConfig(initialization_observations=3))
    assert tracker.step([0.65, 0.65], 0)["state"] is None
    assert tracker.step([0.5, 0.48], 0.1)["state"] is None
    assert tracker.step([0.5, 0.48], 0.2)["state"] is None
    result = tracker.step([0.5, 0.48], 0.3)
    assert result["status"] == "initialized"
    np.testing.assert_allclose(result["state"], [0.5, 0.48, 0.0, 0.0])


def test_partial_measurement_restarts_initialization_window():
    tracker = LanePositionFilter(LaneFilterConfig(initialization_observations=3))
    tracker.step([0.5, 0.48], 0)
    tracker.step([0.5, None], 0.1)
    assert tracker.step([0.5, 0.48], 0.2)["state"] is None
    assert tracker.step([0.5, 0.48], 0.3)["state"] is None
    assert tracker.step([0.5, 0.48], 0.4)["status"] == "initialized"


def test_fresh_experiment_run_reproduces_failure_and_final_artifacts(tmp_path):
    import json
    from rmldnet.kalman_experiments import run

    report = run(tmp_path, seeds=2)
    assert report["verification_seed_start"] == 207
    assert len(report["verification_scenarios"]) == 4
    for name in [
        "initial_stress_results.json",
        "initial_failure_trajectory.json",
        "reacquisition_only_results.json",
        "nominal_trajectory.json",
        "stress_trajectory.json",
    ]:
        assert (tmp_path / name).exists()
    initial = json.loads((tmp_path / "initial_stress_results.json").read_text())
    assert initial["diagnostic_seed_24"]["filtered_rmse_observed"] > 1
    assert report["diagnostic_seed_24"]["filtered_rmse_observed"] < 0.04
