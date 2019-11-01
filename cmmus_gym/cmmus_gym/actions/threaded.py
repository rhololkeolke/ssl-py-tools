import time
from copy import deepcopy
from threading import Event, Lock, Thread
from typing import Any, Dict

import structlog

from cmmus_gym.proto.ssl.radio_pb2 import RobotCommands
from cmmus_gym.proto.ssl.radio_pb2_grpc import RadioStub

from .core import ActionClientStats, RawMovementAction


class ThreadedActionClient(Thread):
    def __init__(self):
        super().__init__()
        self._stats_lock = Lock()
        self._stats = ActionClientStats()

        self._stop_event = Event()

    @property
    def current_action(self) -> Any:
        raise NotImplementedError

    def set_action(self, action: Any) -> Any:
        raise NotImplementedError

    def reset_action(self) -> Any:
        raise NotImplementedError

    def statistics(self) -> ActionClientStats:
        with self._stats_lock:
            return deepcopy(self._stats)

    def stop(self):
        self._stop_event.set()

    def run(self):
        raise NotImplementedError


class RawMovementActionClient(ThreadedActionClient):
    def __init__(self, radio: RadioStub, action_period: float):
        super().__init__()

        # logging setup
        self._log = structlog.get_logger()
        self._log.debug("Creating RawMovementActionClient")

        # properties
        self._radio = radio
        self._action_period = action_period

        # state and threading synchronization
        self._actions_lock = Lock()
        self._actions: Dict[int, RawMovementAction] = {}

    @property
    def action_period(self) -> float:
        return self._action_period

    @property
    def current_action(self) -> Dict[int, RawMovementAction]:
        with self._actions_lock:
            return deepcopy(self._actions)

    def set_action(self, action: RawMovementAction):
        with self._actions_lock:
            self._actions[action.robot_id] = action
        with self._stats_lock:
            self._stats.last_set_action_time = time.time()

    def reset_action(self):
        with self._actions_lock:
            self._actions.clear()
        with self._stats_lock:
            self._stats.last_set_action_time = time.time()

    def run(self):
        self._log.debug("Running raw movement action client thread")
        self._stop_event.clear()

        action_generator = self.__generate_actions()
        command_stream_future = self._radio.CommandStream.future(action_generator)

        while not self._stop_event.is_set() and not command_stream_future.done():
            time.sleep(1)

        if self._stop_event.is_set() and not command_stream_future.done():
            if not command_stream_future.cancel():
                self._log.warn(
                    "thread stop called but cannot cancel"
                    " GRPC command stream client call."
                )
            else:
                self._log.info("Cancelled GRPC command stream client call")

    def __generate_actions(self):
        while not self._stop_event.is_set():
            start_time = time.time()
            action_message = RobotCommands()
            with self._actions_lock:
                for command in self._actions.values():
                    command.to_proto(action_message)
            with self._stats_lock:
                self._stats.num_sent += 1
                self._stats.last_action_sent_time = time.time()
            yield action_message
            elapsed_time = time.time() - start_time

            sleep_time = self._action_period - elapsed_time
            self._log.debug("Sleeping", sleep_time=sleep_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
