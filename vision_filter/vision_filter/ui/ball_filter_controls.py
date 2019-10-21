import imgui
import numpy as np
import structlog

from vision_filter.filter.ball import BasicBallFilter


class BallFilterControls:
    def __init__(self, ball_filter: BasicBallFilter, visible: bool = False):
        self._log = structlog.get_logger()

        self.visible: bool = visible
        self.start_timestamp = 0.0
        self.initial_state = np.zeros(ball_filter.dim_x)
        self.initial_P = np.ones(ball_filter.dim_x)

        self.is_dirty: bool = False

    def __call__(self, ball_filter: BasicBallFilter):
        if not self.visible:
            return

        _, self.visible = imgui.begin("Ball Filter Controls", True)

        changed, value = imgui.input_float("Start timestamp", self.start_timestamp)
        if changed:
            self.is_dirty = True
            self.start_timestamp = value

        self._draw_initial_state()
        self._draw_initial_P()

        imgui.separator()
        changed, value = imgui.input_float(
            "Friction deceleration", ball_filter.friction_decel
        )
        if changed:
            if value >= 0:
                self.is_dirty = True
                ball_filter.friction_decel = value
            else:
                self._log.error(f"Friction deceleration must be >= 0. Got {value}")

        changed, value = imgui.input_float(
            "Friction deadzone", ball_filter.friction_deadzone, format="%e"
        )
        if changed:
            if value >= 0:
                self.is_dirty = True
                ball_filter.friction_deadzone = value
            else:
                self._log.error(f"Friction deadzone must be >= 0. Got {value}")

        changed, value = imgui.input_float(
            "Process Variance", ball_filter.process_variance
        )
        if changed:
            if value >= 0:
                self.is_dirty = True
                ball_filter.process_variance = value
            else:
                self._log.error(f"Process variance must be >= 0. Got {value}")

        self._draw_R(ball_filter)

        imgui.end()

    def _draw_initial_state(self):
        expanded, _ = imgui.collapsing_header("Initial State")
        if expanded:
            changed, values = imgui.input_float2(
                "Position (x, y)", self.initial_state[0], self.initial_state[2]
            )
            if changed:
                self.is_dirty = True
                self.initial_state[0], self.initial_state[2] = values
            changed, values = imgui.input_float2(
                "Velocity (x, y)", self.initial_state[1], self.initial_state[3]
            )
            if changed:
                self.is_dirty = True
                self.initial_state[1], self.initial_state[3] = values

    def _draw_initial_P(self):
        expanded, _ = imgui.collapsing_header("Initial P diagonals")
        if expanded:
            changed, values = imgui.input_float2(
                "Position Uncertainty (x, y)", self.initial_P[0], self.initial_P[2]
            )
            if changed:
                if values[0] >= 0 and values[1] >= 0:
                    self.is_dirty = True
                    self.initial_P[0], self.initial_P[2] = values
                else:
                    self._log.error(f"Position uncertainty must be >= 0. Got {values}")
            changed, values = imgui.input_float2(
                "Velocity Uncertainty (x, y)", self.initial_P[1], self.initial_P[3]
            )
            if changed:
                if values[0] >= 0 and values[1] >= 0:
                    self.is_dirty = True
                    self.initial_P[1], self.initial_P[3] = values
                else:
                    self._log.error(f"Velocity uncertainty must be >= 0. Got {values}")

    def _draw_R(self, ball_filter: BasicBallFilter):
        expanded, _ = imgui.collapsing_header("R")
        if expanded:
            changed, values = imgui.input_float2(
                "R[0, :]", ball_filter.R[0, 0], ball_filter.R[0, 1]
            )
            if changed:
                if values[0] >= 0 and values[1] >= 0:
                    self.is_dirty = True
                    ball_filter.R[0, :] = values

            changed, values = imgui.input_float2(
                "R[1, :]", ball_filter.R[1, 0], ball_filter.R[1, 1]
            )
            if changed:
                if values[0] >= 0 and values[1] >= 0:
                    self.is_dirty = True
                    ball_filter.R[1, :] = values
