from typing import Union

import filterpy.kalman
import numpy as np
import sympy
from filterpy.common import Q_discrete_white_noise


def apply_deadzone(
    value: Union[float, np.ndarray], deadzone: Union[float, np.ndarray]
) -> np.ndarray:
    value = np.array(value)
    value[np.abs(value) < deadzone] = 0

    return value


class BasicBallFilter(filterpy.kalman.KalmanFilter):
    """Filter ball positions/velocities with basic KalmanFilter.    

    dim_x = 4
    dim_z = 2

    X = [x, \dot{x}, y, \dot{y}]
    Z = [x, y]

    The control inputs are:
    u = [\ddot{x}, \ddot{y}];

    These are calculated based on the friction deceleration and the
    current estimated velocities using the equations:

    u[\ddot{x}] = sign(deadzone(X[\dot{x}], friction_deadzone)) * friction_decel

    and similary for u[\ddot{y}].

    This filter assumes ball will decelerate at a constant rate due to
    friction. This ignores slip/no-slip conditions. Also does not deal
    with non-linear effects like bounces off of other robots, walls,
    etc.

    Process noise is based on piecewise white noise model. We are
    assuming that the acceleration is constant for a duration of each
    time period, but the amount differs in an uncorrelated way between
    periods. See
    https://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb#Piecewise-White-Noise-Model
    for more information.

    """

    def __init__(
        self,
        x: np.ndarray = None,
        friction_decel: float = 0.07 * 9806.65,
        friction_deadzone: float = 1e-6,
        P: np.ndarray = None,
        R: np.ndarray = None,
        process_variance: float = 3.5,
        start_timestamp: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(dim_x=4, dim_z=2, *args, **kwargs)
        if friction_decel < 0:
            raise ValueError(
                f"Friction deceleration must be >= 0. Got {friction_decel}"
            )
        self.friction_decel = friction_decel
        self.friction_deadzone = friction_deadzone

        if x is None:
            self.x = np.zeros((4, 1))
        else:
            self.x = x

        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

        if P is None:
            self.P = np.diag([10, 10, 10, 10])
        else:
            self.P = P

        if R is None:
            self.R = np.diag([10, 10])
        else:
            self.R = R

        self._dt = sympy.symbols("\\Delta{t}")

        self.B = sympy.Matrix([[0, 0], [1, 0], [0, 0], [0, 1]])

        # dt and friction decel can vary so F matrix must be
        # recalculated before each predict step
        self.F = np.eye(4)
        self.F[0, 1] = 1
        self.F[2, 3] = 1

        # process noise matrix Q depends on dt Setup a sympy matrix
        # that can be evaluated with specific dt value each update
        self.process_variance = process_variance

        # used to calculate dt parameter
        self.timestamp = 0.0

    def predict(self, dt: float, *args, **kwargs):
        # assuming constant deceleration due to friction loss, as an
        # input to the system
        velocities = self.x[[1, 3]]
        friction_input = (
            np.sign(apply_deadzone(velocities, self.friction_deadzone))
            * self.friction_decel
        )

        if "u" in kwargs:
            kwargs["u"] = friction_input + kwargs["u"]
        else:
            kwargs["u"] = friction_input

        # F, Q, and B depend on dt which is not constant, so calculate
        # new F, Q and B matrices based on current dt.
        self.Q = Q_discrete_white_noise(dim=4, dt=dt, var=self.process_variance)

        self.B[1, 0] = dt
        self.B[3, 1] = dt

        self.F[0, 1] = dt
        self.F[2, 3] = dt

        # update the timestep
        self.timestamp += dt

        # Run the actual prediction step
        return super().predict(*args, **kwargs)
