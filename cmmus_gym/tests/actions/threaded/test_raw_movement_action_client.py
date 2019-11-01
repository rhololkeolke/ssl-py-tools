import contextlib
import time
from concurrent.futures import Future
from functools import partial
from threading import Thread
from typing import List

import mock
import numpy as np
from google.protobuf.empty_pb2 import Empty
from hypothesis import given, settings
from hypothesis.extra import numpy as np_st
from hypothesis.strategies import floats, integers, lists

from cmmus_gym.actions import RawMovementAction
from cmmus_gym.actions.threaded import RawMovementActionClient
from cmmus_gym.proto.ssl import radio_pb2, radio_pb2_grpc


class MockCommandStream(Thread):
    def __init__(self):
        super().__init__()

        self.generator = None
        self.values = []

        self._future = Future()

    def future(self, generator):
        self.generator = generator
        self.start()

        return self._future

    def run(self):
        while not self._future.cancelled():
            try:
                value = next(self.generator)
            except StopIteration:
                break
            self.values.append(value)
        if not self._future.cancelled:
            self._future.set_result(Empty())


class MockRadioClient:
    def __init__(self):
        self.command_streams = []

    def reset(self):
        self.command_streams = []

    @property
    def CommandStream(self):
        command_stream = MockCommandStream()
        self.command_streams.append(command_stream)
        return command_stream


@contextlib.contextmanager
def launch_action_client(action_client, mock_time=True):
    try:
        if mock_time:
            with mock.patch("time.sleep"):
                action_client.start()
                yield action_client
        else:
            action_client.start()
            yield action_client
    finally:
        action_client.stop()
        action_client.join()


@given(
    robot_id=integers(min_value=0, max_value=15),
    wheel_velocities=np_st.arrays(
        np.float32, (4,), floats(min_value=-1, max_value=1, width=32)
    ),
)
def test_set_action_not_running(robot_id, wheel_velocities):
    radio = MockRadioClient()
    action_client = RawMovementActionClient(radio, 1 / 60)

    assert action_client.current_action == {}

    expected_actions = {robot_id: RawMovementAction(robot_id, wheel_velocities)}
    action_client.set_action(expected_actions[robot_id])

    curr_action = action_client.current_action
    assert len(curr_action) == len(expected_actions)
    for robot_id, expected_action in expected_actions.items():
        assert robot_id in curr_action
        action = curr_action[robot_id]
        assert action.robot_id == expected_action.robot_id
        assert np.all(action.wheel_velocities == expected_action.wheel_velocities)


@given(
    robot_id=integers(min_value=0, max_value=15),
    wheel_velocities=np_st.arrays(
        np.float32, (4,), floats(min_value=-1, max_value=1, width=32)
    ),
)
def test_reset_action_not_running(robot_id, wheel_velocities):
    radio = MockRadioClient()
    action_client = RawMovementActionClient(radio, 1 / 60)

    assert action_client.current_action == {}

    expected_actions = {robot_id: RawMovementAction(robot_id, wheel_velocities)}
    action_client.set_action(expected_actions[robot_id])

    curr_action = action_client.current_action
    assert len(curr_action) == len(expected_actions)
    for robot_id, expected_action in expected_actions.items():
        assert robot_id in curr_action
        action = curr_action[robot_id]
        assert action.robot_id == expected_action.robot_id
        assert np.all(action.wheel_velocities == expected_action.wheel_velocities)

    action_client.reset_action()
    assert action_client.current_action == {}


def test_run_with_no_set_action():
    radio = MockRadioClient()
    period = 1 / 60
    action_client = RawMovementActionClient(radio, period)

    assert action_client.current_action == {}
    num_messages = 5
    with launch_action_client(action_client, False):
        time.sleep(period * num_messages)

    stats = action_client.statistics()
    assert stats.num_sent == num_messages
    assert stats.last_action_sent_time is not None

    values = radio.command_streams[0].values
    assert len(values) == num_messages
    for value in values:
        assert value == radio_pb2.RobotCommands()


def test_set_action_running():
    radio = MockRadioClient()
    period = 1 / 60
    action_client = RawMovementActionClient(radio, period)

    first_action = RawMovementAction(0, np.random.randn(4))
    action_client.set_action(first_action)

    second_action = RawMovementAction(0, np.random.randn(4))

    curr_action = action_client.current_action
    assert len(curr_action) == 1
    assert first_action.robot_id in curr_action
    assert first_action.robot_id == curr_action[first_action.robot_id].robot_id
    assert np.all(
        first_action.wheel_velocities
        == curr_action[first_action.robot_id].wheel_velocities
    )

    num_messages = 5
    with launch_action_client(action_client, False):
        time.sleep(period * num_messages)
        action_client.set_action(second_action)
        time.sleep(period * num_messages)

    curr_action = action_client.current_action
    assert len(curr_action) == 1
    assert second_action.robot_id in curr_action
    assert second_action.robot_id == curr_action[second_action.robot_id].robot_id
    assert np.all(
        second_action.wheel_velocities
        == curr_action[second_action.robot_id].wheel_velocities
    )

    stats = action_client.statistics()
    assert stats.num_sent >= num_messages * 2
    assert stats.last_action_sent_time is not None
    assert stats.last_set_action_time is not None

    expected_first_command = radio_pb2.RobotCommands()
    first_action.to_proto(expected_first_command)
    expected_second_command = radio_pb2.RobotCommands()
    second_action.to_proto(expected_second_command)

    values = radio.command_streams[0].values
    assert len(values) >= num_messages * 2
    for i, value in enumerate(values):
        if i < num_messages:
            assert value == expected_first_command
        else:
            assert value == expected_second_command


def test_reset_action_running():
    radio = MockRadioClient()
    period = 1 / 60
    action_client = RawMovementActionClient(radio, period)

    first_action = RawMovementAction(0, np.random.randn(4))
    action_client.set_action(first_action)

    curr_action = action_client.current_action
    assert len(curr_action) == 1
    assert first_action.robot_id in curr_action
    assert first_action.robot_id == curr_action[first_action.robot_id].robot_id
    assert np.all(
        first_action.wheel_velocities
        == curr_action[first_action.robot_id].wheel_velocities
    )

    num_messages = 5
    with launch_action_client(action_client, False):
        time.sleep(period * num_messages)
        action_client.reset_action()
        time.sleep(period * num_messages)

    curr_action = action_client.current_action
    assert curr_action == {}

    stats = action_client.statistics()
    assert stats.num_sent >= num_messages * 2
    assert stats.last_action_sent_time is not None
    assert stats.last_set_action_time is not None

    expected_first_command = radio_pb2.RobotCommands()
    first_action.to_proto(expected_first_command)
    expected_second_command = radio_pb2.RobotCommands()

    values = radio.command_streams[0].values
    assert len(values) >= num_messages * 2
    for i, value in enumerate(values):
        if i < num_messages:
            assert value == expected_first_command
        else:
            assert value == expected_second_command
