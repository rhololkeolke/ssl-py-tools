from functools import partial

import mock
import numpy as np
import pytest
import trio

from cmmus_gym.env import SingleRobotMovementEnv, Team
from vision_filter.proto.messages_robocup_ssl_detection_pb2 import \
    SSL_DetectionFrame


def make_default_detection_frame(camera_id: int):
    frame = SSL_DetectionFrame()
    frame.frame_number = 0
    frame.t_capture = 0
    frame.t_sent = 0
    frame.camera_id = camera_id
    return frame


class MockVisionObs(mock.MagicMock):
    async def run(self):
        return self.sync_run()

    async def clone(self):
        return self.sync_clone()


class MockActionClient(mock.MagicMock):
    async def run(self, period: float):
        return self.sync_run(period)

    async def set_action(self, action):
        return self.sync_set_action(action)

    async def reset_action(self):
        return self.sync_reset_action()


async def start_wrapper(async_fn, *args, task_status=trio.TASK_STATUS_IGNORED):
    task_status.started()
    return await async_fn(*args)


def test_bad_robot_id():
    with pytest.raises(ValueError):
        SingleRobotMovementEnv(
            -1,
            Team.BLUE,
            4,
            -3000,
            -5000,
            None,
            None,
            lambda *args: 0.0,
            lambda *args: False,
        )


def test_bad_num_cameras():
    with pytest.raises(ValueError):
        SingleRobotMovementEnv(
            0,
            Team.BLUE,
            0,
            -3000,
            -5000,
            None,
            None,
            lambda *args: 0.0,
            lambda *args: False,
        )


def test_bad_robot_id_setter():
    env = SingleRobotMovementEnv(
        0,
        Team.BLUE,
        4,
        -3000,
        -5000,
        None,
        None,
        lambda *args: 0.0,
        lambda *args: False,
    )
    with pytest.raises(ValueError):
        env.robot_id = -1


def test_bad_num_cameras_setter():
    env = SingleRobotMovementEnv(
        0,
        Team.BLUE,
        4,
        -3000,
        -5000,
        None,
        None,
        lambda *args: 0.0,
        lambda *args: False,
    )
    with pytest.raises(ValueError):
        env.num_cameras = 0


def test_observation_space():
    env = SingleRobotMovementEnv(
        0,
        Team.BLUE,
        4,
        -3000,
        -5000,
        None,
        None,
        lambda *args: 0.0,
        lambda *args: False,
    )

    example_obs = env.observation_space.sample()
    assert len(example_obs) == 8
    for i in range(env.num_cameras):
        key = f"cam_{i}_detected"
        assert example_obs[key] == 0 or example_obs[key] == 1
    for i in range(env.num_cameras):
        key = f"cam_{i}"
        value = example_obs[key]
        assert value.shape == (3,)


def test_observation_space_updated_on_num_cameras_change():
    env = SingleRobotMovementEnv(
        0,
        Team.BLUE,
        4,
        -3000,
        -5000,
        None,
        None,
        lambda *args: 0.0,
        lambda *args: False,
    )

    example_obs = env.observation_space.sample()
    assert len(example_obs) == 2 * env.num_cameras
    for i in range(env.num_cameras):
        key = f"cam_{i}_detected"
        assert example_obs[key] == 0 or example_obs[key] == 1
    for i in range(env.num_cameras):
        key = f"cam_{i}"
        value = example_obs[key]
        assert value.shape == (3,)

    # now change number of cameras
    env.num_cameras = 1

    example_obs = env.observation_space.sample()
    assert len(example_obs) == 2 * env.num_cameras
    for i in range(env.num_cameras):
        key = f"cam_{i}_detected"
        assert example_obs[key] == 0 or example_obs[key] == 1
    for i in range(env.num_cameras):
        key = f"cam_{i}"
        value = example_obs[key]
        assert value.shape == (3,)


def test_action_space_configured():
    env = SingleRobotMovementEnv(
        0,
        Team.BLUE,
        4,
        -3000,
        -5000,
        None,
        None,
        lambda *args: 0.0,
        lambda *args: False,
    )

    assert np.all(env.action_space.low == -np.ones(4))
    assert np.all(env.action_space.high == np.ones(4))


async def test_reset_when_not_running():
    action_client = MockActionClient()
    vision_obs = MockVisionObs()

    env = SingleRobotMovementEnv(
        0,
        Team.BLUE,
        1,
        -3000,
        -5000,
        vision_obs,
        action_client,
        lambda *args: 0.0,
        lambda *args: False,
    )

    with pytest.raises(RuntimeError):
        await env.reset()


async def test_step_when_not_running():
    action_client = MockActionClient()
    vision_obs = MockVisionObs()

    env = SingleRobotMovementEnv(
        0,
        Team.BLUE,
        1,
        -3000,
        -5000,
        vision_obs,
        action_client,
        lambda *args: 0.0,
        lambda *args: False,
    )

    with pytest.raises(RuntimeError):
        await env.step(env.action_space.sample())


async def test_when_no_observations(nursery):
    action_client = MockActionClient()

    vision_obs = MockVisionObs()
    detections = {}
    vision_obs.sync_clone.return_value = detections

    env = SingleRobotMovementEnv(
        0,
        Team.BLUE,
        1,
        -3000,
        -5000,
        vision_obs,
        action_client,
        lambda *args: 0.0,
        lambda *args: False,
    )

    await nursery.start(partial(start_wrapper, env.run, 1 / 60))

    state = await env.reset()
    assert vision_obs.sync_run.called_once()
    assert action_client.sync_reset_action.called_once()
    assert len(state) == 2
    assert state["cam_0_detected"] == 0
    assert np.all(state["cam_0"] == np.zeros(3))

    # verify curr_state was set
    assert len(env.curr_state) == 2
    assert env.curr_state["cam_0_detected"] == 0
    assert np.all(env.curr_state["cam_0"] == np.zeros(3))

    # call step
    vision_obs.sync_run.reset()
    state, reward, is_terminal, _ = await env.step(env.action_space.sample())
    assert vision_obs.sync_run.called_once()
    assert action_client.sync_set_action.called_once()
    assert reward == 0.0
    assert not is_terminal

    assert len(state) == 2
    assert state["cam_0_detected"] == 0
    assert np.all(state["cam_0"] == np.zeros(3))

    # verify curr_state was set
    assert len(env.curr_state) == 2
    assert env.curr_state["cam_0_detected"] == 0
    assert np.all(env.curr_state["cam_0"] == np.zeros(3))


async def test_with_observations_of_diff_robot(nursery):
    action_client = MockActionClient()

    vision_obs = MockVisionObs()

    detections = {0: make_default_detection_frame(0)}
    blue_robot = detections[0].robots_blue.add()
    blue_robot.robot_id = 1
    blue_robot.x = 0
    blue_robot.y = 0
    blue_robot.pixel_x = 0
    blue_robot.pixel_y = 0
    yellow_robot = detections[0].robots_yellow.add()
    yellow_robot.robot_id = 0
    yellow_robot.x = 0
    yellow_robot.y = 0
    yellow_robot.pixel_x = 0
    yellow_robot.pixel_y = 0

    vision_obs.sync_clone.return_value = detections

    env = SingleRobotMovementEnv(
        0,
        Team.BLUE,
        1,
        -3000,
        -5000,
        vision_obs,
        action_client,
        lambda *args: 0.0,
        lambda *args: False,
    )

    await nursery.start(partial(start_wrapper, env.run, 1 / 60))

    state = await env.reset()
    assert vision_obs.sync_run.called_once()
    assert action_client.sync_reset_action.called_once()
    assert len(state) == 2
    assert state["cam_0_detected"] == 0
    assert np.all(state["cam_0"] == np.zeros(3))

    # verify curr_state was set
    assert len(env.curr_state) == 2
    assert env.curr_state["cam_0_detected"] == 0
    assert np.all(env.curr_state["cam_0"] == np.zeros(3))

    # call step
    vision_obs.sync_run.reset()
    state, reward, is_terminal, _ = await env.step(env.action_space.sample())
    assert vision_obs.sync_run.called_once()
    assert action_client.sync_set_action.called_once()
    assert reward == 0.0
    assert not is_terminal

    assert len(state) == 2
    assert state["cam_0_detected"] == 0
    assert np.all(state["cam_0"] == np.zeros(3))

    # verify curr_state was set
    assert len(env.curr_state) == 2
    assert env.curr_state["cam_0_detected"] == 0
    assert np.all(env.curr_state["cam_0"] == np.zeros(3))


async def test_with_observations_of_expected_robot(nursery):
    action_client = MockActionClient()

    vision_obs = MockVisionObs()

    detections = [
        {0: make_default_detection_frame(0)},
        {0: make_default_detection_frame(0)},
    ]
    # first frame
    blue_robot = detections[0][0].robots_blue.add()
    blue_robot.robot_id = 0
    blue_robot.x = 0
    blue_robot.y = 0
    blue_robot.orientation = 0
    blue_robot.pixel_x = 0
    blue_robot.pixel_y = 0
    # second frame
    blue_robot = detections[1][0].robots_blue.add()
    blue_robot.robot_id = 0
    blue_robot.x = 10
    blue_robot.y = 10
    blue_robot.orientation = np.pi / 2
    blue_robot.pixel_x = 10
    blue_robot.pixel_y = 10

    vision_obs.sync_clone.side_effect = detections

    env = SingleRobotMovementEnv(
        0,
        Team.BLUE,
        1,
        -3000,
        -5000,
        vision_obs,
        action_client,
        lambda *args: 0.0,
        lambda *args: False,
    )

    await nursery.start(partial(start_wrapper, env.run, 1 / 60))

    state = await env.reset()
    assert vision_obs.sync_run.called_once()
    assert action_client.sync_reset_action.called_once()
    assert len(state) == 2
    assert state["cam_0_detected"] == 1
    assert np.all(state["cam_0"] == np.zeros(3))

    # verify curr_state was set
    assert len(env.curr_state) == 2
    assert env.curr_state["cam_0_detected"] == 1
    assert np.all(env.curr_state["cam_0"] == np.zeros(3))

    # call step
    vision_obs.sync_run.reset()
    state, reward, is_terminal, _ = await env.step(env.action_space.sample())
    assert vision_obs.sync_run.called_once()
    assert action_client.sync_set_action.called_once()
    assert reward == 0.0
    assert not is_terminal

    assert len(state) == 2
    assert state["cam_0_detected"] == 1
    assert np.all(state["cam_0"] == np.array([10, 10, np.pi / 2]))

    # verify curr_state was set
    assert len(env.curr_state) == 2
    assert env.curr_state["cam_0_detected"] == 1
    assert np.all(env.curr_state["cam_0"] == np.array([10, 10, np.pi / 2]))
