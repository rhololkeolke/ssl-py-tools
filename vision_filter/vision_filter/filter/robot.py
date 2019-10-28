from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import structlog
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter


def angle_mod(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Return angle within [-pi, pi].

    Based on answer stackoverflow answer to [1]_.

    .. [1] 'C: How to wrap a float to the interval [-pi, pi]'
       https://stackoverflow.com/a/29871193

    """
    return -np.pi + np.fmod(2 * np.pi + np.fmod(angle + np.pi, 2 * np.pi), 2 * np.pi)


# These values are taken from soccer robot_tracker(.cc|.h)
@dataclass
class RobotFilterSettings:
    position_variance: float
    velocity_variance: float
    angvel_variance: float

    theta_variance: float = np.radians(0.5) ** 2
    confidence_threshold: float = 0.1
    no_data_timeout: float = 10.0


@dataclass
class UnknownRobotFilterSettings(RobotFilterSettings):
    position_variance: float = 10 ** 2
    velocity_variance: float = 50 ** 2
    angvel_variance: float = np.radians(80.0) ** 2


@dataclass
class KnownRobotFilterSettings(RobotFilterSettings):
    position_variance: float = 5 ** 2
    velocity_variance: float = 200.0 ** 2
    angvel_variance: float = np.radians(70.0) ** 2


class RobotFilter(UnscentedKalmanFilter):
    def __init__(
        self,
        filter_settings: RobotFilterSettings,
        dt: float = 1 / 60,
        points: Optional[Any] = None,
    ):
        self._log = structlog.get_logger().bind(dt=dt, points=points)
        self._settings = filter_settings
        super().__init__(
            dim_x=6,
            dim_z=3,
            dt=dt,
            hx=self._hx,
            fx=self._fx,
            points=points or MerweScaledSigmaPoints(4, alpha=0.1, beta=2.0, kappa=-1),
            x_mean_fn=self._x_mean_fn,
            z_mean_fn=self._z_mean_fn,
            residual_x=self._residual_x,
            residual_z=self._residual_z,
        )

        # set initial state to all zeros, and initialize the state
        # covariances
        self.set_state(np.zeros((6, 1)))

        # TODO(dschwab): taken from robot tracker. I'm not sure this
        # is 100% correct or at least 100% the best it can be. Seems
        # to be assuming that given a perfect velocity, our process
        # will give us a perfect position. At the least this does not
        # account for bad friction models, pushing, slippage, etc
        # where velocity integration over the timestep is not 100%
        # correct.
        self.Q = np.diag(
            [
                0,
                0,
                0,
                self._settings.velocity_variance,
                self._settings.velocity_variance,
                self._settings.angvel_variance,
            ]
        )

        # TODO(dschwab): taken from robot tracker. I'm not sure this
        # is 100% correct or at least 100% the best it can be.
        self.R = np.diag(
            [
                self._settings.position_variance,
                self._settings.position_variance,
                self._settings.theta_variance,
            ]
        )

    def set_state(self, state: np.ndarray, P: Optional[np.ndarray] = None):
        """Manually set the state and state covariance.

        This is useful to initialize the filter on creation, as well
        as to reinitialize when the robot has come back in after a
        period of non-detection.

        Parameters
        ----------
        state: array
          Sets the current filter state to this value. Must have shape (6,1).
        P: array, optional
          Sets current state covariance to this value. Must have shape
          (6, 6). Typically, you would set the diagonal using
          something like `np.diag`.

        """
        self.x = state
        # initialize the variance
        if P:
            self.P = P
        else:
            self.P = np.diag(
                [
                    self._settings.position_variance,
                    self._setings.position_variance,
                    self._settings.theta_variance,
                    0,
                    0,
                    0,
                ]
            )

    def _hx(self, state):
        # converts state from [x, y, theta, vx, vy, omega] to [x, y,
        # theta]
        return np.array([state[0, 0], state[1, 0], state[2, 0]])

    def _fx(self, state, dt):
        x = state[0, 0]
        y = state[1, 0]
        theta = state[2, 0]
        vx = state[3, 0]
        vy = state[4, 0]
        omega = state[5, 0]

        # TODO(dschwab): add support for robot commands
        # For now assume that velocities are constant
        theta = angle_mod(theta + dt * omega)
        x += dt * vx
        y += dt * vy

        # TODO(dschwab): add boundary checks that zero out velocities

        # copy back to state
        state[0, 0] = x
        state[1, 0] = y
        state[2, 0] = theta
        # in future implementation velocities may change due to
        # collisions with walls, so copy them back too
        state[3, 0] = vx
        state[4, 0] = vy
        state[5, 0] = omega

    def _x_mean_fn(self, sigmas, Wm):
        state = np.zeros((6, len(sigmas)))

        ang_components = np.zeros((2, len(sigmas)))
        for i, (sigma, w) in enumerate(zip(sigmas, Wm)):
            state[:, i] = sigma * w
            ang_components[0, i] = np.sin(sigma[2]) * w
            ang_components[1, i] = np.cos(sigma[2]) * w

        avg_state = np.mean(state, axis=1)
        avg_ang_components = np.mean(ang_components, axis=1)

        avg_state[2] = np.atan2(avg_ang_components[0], avg_ang_components[1])
        return avg_state

    def _z_mean_fn(self, sigmas, Wm):
        meas = np.zeros((3, len(sigmas)))

        ang_components = np.zeros((2, len(sigmas)))
        for i, (sigma, w) in enumerate(zip(sigmas, Wm)):
            meas[:, i] = sigma * w
            ang_components[0, i] = np.sin(sigma[2]) * w
            ang_components[1, i] = np.cos(sigma[2]) * w

        avg_meas = np.mean(meas, axis=1)
        avg_ang_components = np.mean(ang_components, axis=1)

        avg_meas[2] = np.atan2(avg_ang_components[0], avg_ang_components[1])
        return avg_meas

    def _residual_x(self, a, b):
        y = a - b
        # angles need to support wrap.
        #
        # TODO(dschwab): verify that this will always give correct
        # output
        y[2] = angle_mod(y[2])
        return y

    def _residual_z(self, a, b):
        y = a - b
        # angles need to support wrap.
        #
        # TODO(dschwab): verify that this will always give correct
        # output
        y[2] = angle_mod(y[2])
        return y
