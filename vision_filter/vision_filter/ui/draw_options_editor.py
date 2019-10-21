import math
from typing import List

import imgui
import structlog


class DrawOptionsEditor:
    def __init__(self, visible: bool = False):
        self._log = structlog.get_logger()
        self.visible: bool = visible
        self.draw_ball_detections_discrete: bool = True
        self.ball_detection_discrete_color: List[float] = [1.0, 0.65, 0.0, 1.0]

        self.draw_ball_filter_discrete: bool = True
        self.ball_filter_discrete_color: List[float] = [0.416, 0.353, 0.804, 1.0]

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
                changed, self.ball_detection_discrete_color = imgui.color_edit4(
                    "Discrete ball color", *self.ball_detection_discrete_color
                )

        expanded, _ = imgui.collapsing_header("Ball Filter Options")
        if expanded:
            _, self.draw_ball_filter_discrete = imgui.checkbox(
                "Draw discrete", self.draw_ball_filter_discrete
            )
            if self.draw_ball_filter_discrete:
                changed, self.ball_filter_discrete_color = imgui.color_edit4(
                    "Discrete ball color", *self.ball_filter_discrete_color
                )

        imgui.end()
