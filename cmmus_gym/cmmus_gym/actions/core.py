"""Common classes used by different ActionClients."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from cmmus_gym.proto.ssl.radio_pb2 import RobotCommands


@dataclass
class ActionClientStats:
    num_sent: int = 0
    last_set_action_time: Optional[float] = None
    last_action_sent_time: Optional[float] = None


@dataclass
class RawMovementAction:
    robot_id: int
    # Should be in range [-1, 1]. They will be scaled to the
    # appropriate radio commands automatically
    wheel_velocities: np.ndarray

    def to_proto(self, message: RobotCommands):
        robot_command = message.commands[self.robot_id]

        wheel_velocities = np.clip(127 * self.wheel_velocities, -127, 127).astype(
            np.int8
        )
        robot_command.wheel_velocity[:] = wheel_velocities
