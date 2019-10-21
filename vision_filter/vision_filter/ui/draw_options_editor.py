import math
from typing import List

import imgui
import structlog


class DrawOptionsEditor:
    def __init__(self, detected_ball, filtered_ball, visible: bool = False):
        self._log = structlog.get_logger()
        self.visible: bool = visible
        self.draw_ball_detections_discrete: bool = True
        self.detected_ball = detected_ball

        self.draw_ball_filter_discrete: bool = True
        self.filtered_ball = filtered_ball

    def __call__(self):
        if not self.visible:
            return

        _, self.visible = imgui.begin("Drawing Options", True)

        expanded, _ = imgui.collapsing_header("Ball Detection Options")
        if expanded:
            _, self.draw_ball_detections_discrete = imgui.checkbox(
                "Draw discrete", self.draw_ball_detections_discrete
            )
            if self.draw_ball_detections_discrete:
                changed, value = imgui.color_edit4(
                    "Discrete ball color", *self.detected_ball.color
                )
                if changed:
                    self.detected_ball.color = value

        expanded, _ = imgui.collapsing_header("Ball Filter Options")
        if expanded:
            _, self.draw_ball_filter_discrete = imgui.checkbox(
                "Draw discrete", self.draw_ball_filter_discrete
            )
            if self.draw_ball_filter_discrete:
                changed, value = imgui.color_edit4(
                    "Discrete ball color", *self.filtered_ball.color
                )
                if changed:
                    self.filtered_ball.color = value

        imgui.end()
