"""Asynchronously send commands to robot."""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import structlog
import trio
import trio_util

from cmmus_gym.proto.ssl.radio_grpc import RadioStub
from cmmus_gym.proto.ssl.radio_pb2 import RobotCommands


@dataclass
class ActionClientStats:
    num_sent: int = 0
    last_set_action_time: Optional[float] = None
    last_action_sent_time: Optional[float] = None


class ActionClient:
    """Asynchronously send actions to robot(s)."""

    async def run(self, period: float):
        """Send latest command at specified period.

        Generally should be launched with nursery.start_soon
        """
        raise NotImplementedError

    async def set_action(self, action: Any) -> Any:
        """Set the action to send at the next period.

        If the action does some kind of filtering, then it may be
        necessary to return some additional state information. For
        example, a low-pass filter on the actions may require that the
        agent has access to the filter state. So set_action should
        return the filtered state.

        """
        raise NotImplementedError

    async def reset_action(self):
        """Cancels any commands being sent."""
        raise NotImplementedError

    def statistics(self) -> Any:
        """Useful data about the actions being sent.

        Typically includes things like how many actions have been
        sent, time of last action being set, etc.

        """
        raise NotImplementedError


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


class RawMovementActionClient(ActionClient):
    """Sends raw movement commands to a single robot."""

    def __init__(self, radio: RadioStub):
        self._log = structlog.get_logger().bind(radio=radio)

        # this is what communicates with the radio
        self.radio = radio

        # storage of actions
        self._action_message_lock = trio.Lock()
        self._action_message = RobotCommands()

        # track info about how this client is used
        self._stats: ActionClientStats = ActionClientStats()

    async def run(self, period: float):
        """Send messages to radio server at specified rate."""
        async with self.radio.CommandStream.open() as stream:
            async for elapsed, delta in trio_util.periodic(period):
                self._log.debug("Sending next action", elapsed=elapsed, delta=delta)
                async with self._action_message_lock:
                    await stream.send_message(self._action_message)
                self._stats.last_action_sent_time = trio.current_time()

    async def set_action(self, action: RawMovementAction):
        self._log.debug("Setting action", action=action)
        async with self._action_message_lock:
            action.to_proto(self._action_message)
        self._stats.last_set_action_time = trio.current_time()

    async def reset_action(self):
        self._log.debug("Resetting actions")
        async with self._action_message_lock:
            self._action_message.Clear()
        self._stats.last_set_action_time = trio.current_time()

    def statistics(self) -> ActionClientStats:
        return self._stats
