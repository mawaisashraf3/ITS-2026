"""Validated linear Gaussian filtering and two-anchor lane-position dynamics."""

from dataclasses import dataclass
import numpy as np


def covariance(value, dimension: int, name: str, positive_definite=False):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (dimension, dimension) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite {dimension}x{dimension} matrix")
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-9):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(matrix)
    tolerance = 1e-12 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if eigenvalues.min() < -tolerance or (positive_definite and eigenvalues.min() <= 0):
        raise ValueError(
            f"{name} must be positive "
            + ("definite" if positive_definite else "semidefinite")
        )
    return (matrix + matrix.T) / 2


@dataclass(frozen=True)
class UpdateResult:
    accepted: bool
    innovation: np.ndarray
    innovation_covariance: np.ndarray
    gain: np.ndarray
    normalized_innovation_squared: float


class LinearKalmanFilter:
    """F/H/Q/R linear filter, optional control input and innovation gating.

    State order and units belong to the caller. Measurements must be finite;
    missing observations are handled by prediction without an update.
    """

    def __init__(self, state, P, F, H, Q, R, B=None):
        self._x = np.asarray(state, dtype=float).copy()
        if self._x.ndim != 1 or not len(self._x) or not np.isfinite(self._x).all():
            raise ValueError("State must be a finite vector")
        self.n = len(self._x)
        self.F = self._matrix(F, (self.n, self.n), "F")
        self.H = np.asarray(H, dtype=float)
        if (
            self.H.ndim != 2
            or self.H.shape[1] != self.n
            or not len(self.H)
            or not np.isfinite(self.H).all()
        ):
            raise ValueError("Invalid H")
        self.P = covariance(P, self.n, "P")
        self.Q = covariance(Q, self.n, "Q")
        self.R = covariance(R, len(self.H), "R", True)
        self.B = None if B is None else np.asarray(B, dtype=float)
        if self.B is not None and (
            self.B.ndim != 2
            or self.B.shape[0] != self.n
            or not np.isfinite(self.B).all()
        ):
            raise ValueError("Invalid control matrix B")

    @staticmethod
    def _matrix(value, shape, name):
        value = np.asarray(value, dtype=float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid {name} shape or values")
        return value.copy()

    @property
    def state(self):
        return self._x.copy()

    def predict(self, F=None, Q=None, control=None):
        transition = self.F if F is None else self._matrix(F, (self.n, self.n), "F")
        process = self.Q if Q is None else covariance(Q, self.n, "Q")
        state = transition @ self._x
        if control is not None:
            if self.B is None:
                raise ValueError("Control input requires B")
            u = np.asarray(control, dtype=float)
            if u.shape != (self.B.shape[1],) or not np.isfinite(u).all():
                raise ValueError("Invalid control input")
            state += self.B @ u
        P = transition @ self.P @ transition.T + process
        if not np.isfinite(state).all():
            raise FloatingPointError("Nonfinite predicted state")
        self._x = state
        self.P = covariance(P, self.n, "predicted P")
        return self.state

    def update(self, measurement, H=None, R=None, gate=None):
        observation = self.H if H is None else np.asarray(H, dtype=float)
        if (
            observation.ndim != 2
            or observation.shape[1] != self.n
            or not len(observation)
            or not np.isfinite(observation).all()
        ):
            raise ValueError("Invalid observation matrix")
        noise = self.R if R is None else covariance(R, len(observation), "R", True)
        if noise.shape != (len(observation), len(observation)):
            raise ValueError("R does not match observation dimensions")
        z = np.asarray(measurement, dtype=float)
        if z.shape != (len(observation),) or not np.isfinite(z).all():
            raise ValueError("Invalid measurement")
        if gate is not None and (not np.isfinite(gate) or gate <= 0):
            raise ValueError("Gate must be positive")
        innovation = z - observation @ self._x
        S = observation @ self.P @ observation.T + noise
        try:
            chol = np.linalg.cholesky(S)
            whitened = np.linalg.solve(chol, innovation)
            gain = np.linalg.solve(S, (self.P @ observation.T).T).T
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "Innovation covariance is not positive definite"
            ) from error
        nis = float(whitened @ whitened)
        accepted = gate is None or nis <= gate
        if accepted:
            x = self._x + gain @ innovation
            complement = np.eye(self.n) - gain @ observation
            P = complement @ self.P @ complement.T + gain @ noise @ gain.T
            if not np.isfinite(x).all():
                raise FloatingPointError("Nonfinite updated state")
            self._x = x
            self.P = covariance(P, self.n, "updated P")
        return UpdateResult(accepted, innovation, S, gain, nis)


@dataclass(frozen=True)
class LaneFilterConfig:
    acceleration_variance: float = 0.0004
    measurement_variance: float = 0.000144
    initial_velocity_variance: float = 0.01
    gate: float | None = 9.21
    initialization_observations: int = 5
    reacquire_after: int | None = 8
    reacquire_max_spread: float = 0.06

    def __post_init__(self):
        for name in [
            "acceleration_variance",
            "measurement_variance",
            "initial_velocity_variance",
        ]:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.gate is not None and (not np.isfinite(self.gate) or self.gate <= 0):
            raise ValueError("Invalid innovation gate")
        if (
            type(self.initialization_observations) is not int
            or self.initialization_observations < 1
        ):
            raise ValueError("initialization_observations must be positive")
        if self.reacquire_after is not None and (
            type(self.reacquire_after) is not int or self.reacquire_after < 3
        ):
            raise ValueError("reacquire_after must be at least 3 or None")
        if not np.isfinite(self.reacquire_max_spread) or self.reacquire_max_spread <= 0:
            raise ValueError("Invalid reacquisition spread")


def lane_dynamics(dt: float, acceleration_variance: float):
    """Discrete random-acceleration model: [near, far, v_near, v_far]."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive")
    if not np.isfinite(acceleration_variance) or acceleration_variance < 0:
        raise ValueError("Invalid acceleration variance")
    F = np.eye(4)
    F[:2, 2:] = np.eye(2) * dt
    G = np.vstack([np.eye(2) * (0.5 * dt * dt), np.eye(2) * dt])
    return F, acceleration_variance * (G @ G.T)


class LanePositionFilter:
    """Filter mask-derived near/far corridor centers in normalized image width.

    Initializing requires a coherent window of complete anchor pairs; partial observations are allowed afterward.
    Timestamps must increase within a single verified or controlled sequence.
    """

    def __init__(self, config=None):
        self.config = config or LaneFilterConfig()
        self.filter = None
        self.timestamp = None
        self.rejected_window = []
        self.initialization_window = []

    def reset(self):
        self.filter = None
        self.timestamp = None
        self.rejected_window = []
        self.initialization_window = []

    def _initialize(self, centers):
        H = np.hstack([np.eye(2), np.zeros((2, 2))])
        R = np.eye(2) * self.config.measurement_variance
        self.filter = LinearKalmanFilter(
            np.r_[centers, [0.0, 0.0]],
            np.diag(
                [self.config.measurement_variance] * 2
                + [self.config.initial_velocity_variance] * 2
            ),
            np.eye(4),
            H,
            np.zeros((4, 4)),
            R,
        )

    def step(self, centers, timestamp):
        if not np.isfinite(timestamp) or (
            self.timestamp is not None and timestamp <= self.timestamp
        ):
            raise ValueError("Timestamps must be finite and strictly increasing")
        z = np.array([np.nan if x is None else x for x in centers], dtype=float)
        if (
            z.shape != (2,)
            or np.isinf(z).any()
            or np.any((z[np.isfinite(z)] < 0) | (z[np.isfinite(z)] > 1))
        ):
            raise ValueError("Centers must be normalized coordinates or missing")
        observed = np.isfinite(z)
        previous = self.timestamp
        self.timestamp = float(timestamp)
        if self.filter is None:
            if observed.all():
                self.initialization_window.append(z.copy())
                self.initialization_window = self.initialization_window[
                    -self.config.initialization_observations :
                ]
            else:
                self.initialization_window = []
            ready = (
                len(self.initialization_window)
                == self.config.initialization_observations
            )
            if ready:
                ready = bool(
                    np.all(
                        np.ptp(self.initialization_window, axis=0)
                        <= self.config.reacquire_max_spread
                    )
                )
            if not ready:
                return {
                    "status": "waiting_for_stable_initialization",
                    "measurement": [float(v) if np.isfinite(v) else None for v in z],
                    "state": None,
                    "predicted_state": None,
                    "covariance": None,
                    "accepted": False,
                    "nis": None,
                    "reacquired": False,
                }
            self._initialize(np.median(self.initialization_window, axis=0))
            predicted = self.filter.state
            prior = self.filter.P.copy()
            result = None
            status = "initialized"
            accepted = True
        else:
            F, Q = lane_dynamics(
                timestamp - previous, self.config.acceleration_variance
            )
            predicted = self.filter.predict(F, Q)
            prior = self.filter.P.copy()
            result = None
            accepted = False
            status = "prediction_only"
            if observed.any():
                H = np.hstack([np.eye(2), np.zeros((2, 2))])[observed]
                R = np.eye(observed.sum()) * self.config.measurement_variance
                result = self.filter.update(z[observed], H, R, self.config.gate)
                accepted = result.accepted
                status = "updated" if accepted else "innovation_rejected"
                if (
                    not accepted
                    and observed.all()
                    and self.config.reacquire_after is not None
                ):
                    self.rejected_window.append(z.copy())
                    self.rejected_window = self.rejected_window[
                        -self.config.reacquire_after :
                    ]
                    if len(self.rejected_window) == self.config.reacquire_after:
                        window = np.asarray(self.rejected_window)
                        if np.all(
                            np.ptp(window, axis=0) <= self.config.reacquire_max_spread
                        ):
                            self._initialize(np.median(window, axis=0))
                            self.rejected_window = []
                            status = "reacquired"
                else:
                    self.rejected_window = []
            else:
                self.rejected_window = []
        return {
            "status": status,
            "measurement": [float(v) if np.isfinite(v) else None for v in z],
            "state": self.filter.state.tolist(),
            "predicted_state": predicted.tolist(),
            "covariance": self.filter.P.tolist(),
            "predicted_covariance": prior.tolist(),
            "accepted": accepted,
            "reacquired": status == "reacquired",
            "nis": None if result is None else result.normalized_innovation_squared,
            "gain": None if result is None else result.gain.tolist(),
        }
