from functools import partial

import mock
import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import floats, integers, just, one_of

from cmmus_gym.actions import RawMovementAction
from cmmus_gym.env import Team
from cmmus_gym.env.threaded import SingleRobotRawMovementEnv
from vision_filter.proto.messages_robocup_ssl_detection_pb2 import \
    SSL_DetectionFrame


def make_default_detection_frame(camera_id: int):
    frame = SSL_DetectionFrame()
    frame.frame_number = 0
    frame.t_capture = 0
    frame.t_sent = 0
    frame.camera_id = camera_id
    return frame


def start_side_effect(mock_obj):
    mock_obj.is_alive.return_value = True


def stop_side_effect(mock_obj):
    mock_obj.is_alive.return_value = False


@pytest.fixture
def mock_observations():
    obs = mock.MagicMock()
    obs.get_latest.return_value = {}
    obs.is_alive.return_value = False
    obs.start.side_effect = partial(start_side_effect, obs)
    obs.stop.side_effect = partial(stop_side_effect, obs)
    return obs


@pytest.fixture
def mock_action_client():
    action_client = mock.MagicMock()
    action_client.current_action = mock.PropertyMock()
    action_client.current_action.return_value = {}
    action_client.is_alive.return_value = False
    action_client.start.side_effect = partial(start_side_effect, action_client)
    action_client.stop.side_effect = partial(stop_side_effect, action_client)
    return action_client


def dummy_reward(*args):
    return 0.0


def dummy_is_terminal(*args):
    return False


@given(bad_robot_id=integers(max_value=-1))
def test_bad_robot_id(bad_robot_id, mock_observations, mock_action_client):
    with pytest.raises(ValueError):
        SingleRobotRawMovementEnv(
            bad_robot_id,
            Team.BLUE,
            4,
            3000,
            5000,
            mock_observations,
            mock_action_client,
            dummy_reward,
            dummy_is_terminal,
        )

    env = SingleRobotRawMovementEnv(
        0,
        Team.BLUE,
        4,
        3000,
        5000,
        mock_observations,
        mock_action_client,
        dummy_reward,
        dummy_is_terminal,
    )
    with pytest.raises(ValueError):
        env.robot_id = bad_robot_id


@given(bad_num_cameras=integers(max_value=0))
def test_bad_num_cameras(bad_num_cameras, mock_observations, mock_action_client):
    with pytest.raises(ValueError):
        SingleRobotRawMovementEnv(
            0,
            Team.BLUE,
            bad_num_cameras,
            3000,
            5000,
            mock_observations,
            mock_action_client,
            dummy_reward,
            dummy_is_terminal,
        )

    env = SingleRobotRawMovementEnv(
        0,
        Team.BLUE,
        4,
        3000,
        5000,
        mock_observations,
        mock_action_client,
        dummy_reward,
        dummy_is_terminal,
    )
    with pytest.raises(ValueError):
        env.num_cameras = bad_num_cameras


@given(bad_field_width=one_of(just(0.0), floats(max_value=0)))
def test_bad_field_width(bad_field_width, mock_observations, mock_action_client):
    with pytest.raises(ValueError):
        SingleRobotRawMovementEnv(
            0,
            Team.BLUE,
            4,
            bad_field_width,
            5000,
            mock_observations,
            mock_action_client,
            dummy_reward,
            dummy_is_terminal,
        )

    env = SingleRobotRawMovementEnv(
        0,
        Team.BLUE,
        4,
        3000,
        5000,
        mock_observations,
        mock_action_client,
        dummy_reward,
        dummy_is_terminal,
    )
    with pytest.raises(ValueError):
        env.field_width = bad_field_width


@given(bad_field_length=one_of(just(0.0), floats(max_value=0)))
def test_bad_field_length(bad_field_length, mock_observations, mock_action_client):
    with pytest.raises(ValueError):
        SingleRobotRawMovementEnv(
            0,
            Team.BLUE,
            4,
            5000,
            bad_field_length,
            mock_observations,
            mock_action_client,
            dummy_reward,
            dummy_is_terminal,
        )

    env = SingleRobotRawMovementEnv(
        0,
        Team.BLUE,
        4,
        3000,
        5000,
        mock_observations,
        mock_action_client,
        dummy_reward,
        dummy_is_terminal,
    )
    with pytest.raises(ValueError):
        env.field_length = bad_field_length


def test_is_running_property(mock_observations, mock_action_client):
    env = SingleRobotRawMovementEnv(
        0,
        Team.BLUE,
        4,
        3000,
        5000,
        mock_observations,
        mock_action_client,
        dummy_reward,
        dummy_is_terminal,
    )

    assert not mock_observations.is_alive()
    assert not mock_action_client.is_alive()
    assert not env.is_running

    env.start()

    assert mock_observations.is_alive()
    assert mock_action_client.is_alive()
    assert env.is_running

    env.close()

    assert not mock_observations.is_alive()
    assert not mock_action_client.is_alive()
    assert not env.is_running

    env.start()

    assert mock_observations.is_alive()
    assert mock_action_client.is_alive()
    assert env.is_running

    mock_observations.stop()

    assert not mock_observations.is_alive()
    assert mock_action_client.is_alive()
    assert not env.is_running

    mock_observations.start()
    mock_action_client.stop()

    assert mock_observations.is_alive()
    assert not mock_action_client.is_alive()
    assert not env.is_running


@given(num_cameras=integers(min_value=1, max_value=4))
def test_observation_space(num_cameras, mock_observations, mock_action_client):
    env = SingleRobotRawMovementEnv(
        0,
        Team.BLUE,
        num_cameras,
        3000,
        5000,
        mock_observations,
        mock_action_client,
        dummy_reward,
        dummy_is_terminal,
    )
    example_obs = env.observation_space.sample()
    assert len(example_obs) == 2 * num_cameras
    for i in range(num_cameras):
        key = f"cam_{i}_detected"
        assert example_obs[key] == 0 or example_obs[key] == 1
    for i in range(num_cameras):
        key = f"cam_{i}"
        value = example_obs[key]
        assert value.shape == (3,)


def test_observation_space_updated_on_num_cameras_change(
    mock_observations, mock_action_client
):
    env = SingleRobotRawMovementEnv(
        0,
        Team.BLUE,
        1,
        3000,
        5000,
        mock_observations,
        mock_action_client,
        dummy_reward,
        dummy_is_terminal,
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

    # change number of cameras
    env.num_cameras = 2

    example_obs = env.observation_space.sample()
    assert len(example_obs) == 2 * env.num_cameras
    for i in range(env.num_cameras):
        key = f"cam_{i}_detected"
        assert example_obs[key] == 0 or example_obs[key] == 1
    for i in range(env.num_cameras):
        key = f"cam_{i}"
        value = example_obs[key]
        assert value.shape == (3,)


def test_action_space_configured(mock_observations, mock_action_client):
    env = SingleRobotRawMovementEnv(
        0,
        Team.BLUE,
        1,
        3000,
        5000,
        mock_observations,
        mock_action_client,
        dummy_reward,
        dummy_is_terminal,
    )

    assert np.all(env.action_space.low == -np.ones(4))
    assert np.all(env.action_space.high == np.ones(4))


def test_reset_when_not_running(mock_observations, mock_action_client):
    env = SingleRobotRawMovementEnv(
        0,
        Team.BLUE,
        1,
        3000,
        5000,
        mock_observations,
        mock_action_client,
        dummy_reward,
        dummy_is_terminal,
    )

    assert not env.is_running

    with pytest.raises(RuntimeError):
        env.reset()


def test_step_when_not_running(mock_observations, mock_action_client):
    env = SingleRobotRawMovementEnv(
        0,
        Team.BLUE,
        1,
        3000,
        5000,
        mock_observations,
        mock_action_client,
        dummy_reward,
        dummy_is_terminal,
    )

    assert not env.is_running

    with pytest.raises(RuntimeError):
        env.step(env.action_space.sample())


def test_close_when_not_running(mock_observations, mock_action_client):
    env = SingleRobotRawMovementEnv(
        0,
        Team.BLUE,
        1,
        3000,
        5000,
        mock_observations,
        mock_action_client,
        dummy_reward,
        dummy_is_terminal,
    )

    assert not env.is_running

    env.close()

    mock_observations.join.assert_not_called()
    mock_action_client.join.assert_not_called()


def test_no_observations(mock_observations, mock_action_client):
    env = SingleRobotRawMovementEnv(
        0,
        Team.BLUE,
        1,
        3000,
        5000,
        mock_observations,
        mock_action_client,
        dummy_reward,
        dummy_is_terminal,
    )

    env.start()

    assert env.is_running

    state = env.reset()
    mock_observations.get_latest.assert_called_once()
    mock_action_client.reset_action.assert_called_once()
    assert len(state) == 2
    assert state["cam_0_detected"] == 0
    assert np.all(state["cam_0"] == np.zeros(3))

    assert len(env.curr_state) == 2
    assert env.curr_state["cam_0_detected"] == 0
    assert np.all(env.curr_state["cam_0"] == np.zeros(3))

    # reset mocks
    mock_observations.reset_mock()

    # call step
    raw_action = env.action_space.sample()
    action = RawMovementAction(env.robot_id, raw_action)
    state, _, _, _ = env.step(raw_action)
    mock_observations.get_latest.assert_called_once()
    assert len(mock_action_client.set_action.mock_calls) == 1
    assert (
        mock_action_client.set_action.mock_calls[0].args[0].robot_id == action.robot_id
    )
    assert np.all(
        mock_action_client.set_action.mock_calls[0].args[0].wheel_velocities
        == action.wheel_velocities
    )

    assert len(state) == 2
    assert state["cam_0_detected"] == 0
    assert np.all(state["cam_0"] == np.zeros(3))

    assert len(env.curr_state) == 2
    assert env.curr_state["cam_0_detected"] == 0
    assert np.all(env.curr_state["cam_0"] == np.zeros(3))


def test_observations_of_diff_robot(mock_observations, mock_action_client):
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

    mock_observations.get_latest.return_value = detections

    env = SingleRobotRawMovementEnv(
        0,
        Team.BLUE,
        1,
        3000,
        5000,
        mock_observations,
        mock_action_client,
        dummy_reward,
        dummy_is_terminal,
    )

    env.start()

    assert env.is_running

    state = env.reset()
    mock_observations.get_latest.assert_called_once()
    mock_action_client.reset_action.assert_called_once()
    assert len(state) == 2
    assert state["cam_0_detected"] == 0
    assert np.all(state["cam_0"] == np.zeros(3))

    assert len(env.curr_state) == 2
    assert env.curr_state["cam_0_detected"] == 0
    assert np.all(env.curr_state["cam_0"] == np.zeros(3))

    # reset mocks
    mock_observations.reset_mock()

    # call step
    raw_action = env.action_space.sample()
    action = RawMovementAction(env.robot_id, raw_action)
    state, _, _, _ = env.step(raw_action)
    mock_observations.get_latest.assert_called_once()
    assert len(mock_action_client.set_action.mock_calls) == 1
    assert (
        mock_action_client.set_action.mock_calls[0].args[0].robot_id == action.robot_id
    )
    assert np.all(
        mock_action_client.set_action.mock_calls[0].args[0].wheel_velocities
        == action.wheel_velocities
    )

    assert len(state) == 2
    assert state["cam_0_detected"] == 0
    assert np.all(state["cam_0"] == np.zeros(3))

    assert len(env.curr_state) == 2
    assert env.curr_state["cam_0_detected"] == 0
    assert np.all(env.curr_state["cam_0"] == np.zeros(3))


def test_observations_of_expected_robot(mock_observations, mock_action_client):
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

    mock_observations.get_latest.side_effect = detections

    env = SingleRobotRawMovementEnv(
        0,
        Team.BLUE,
        1,
        3000,
        5000,
        mock_observations,
        mock_action_client,
        dummy_reward,
        dummy_is_terminal,
    )

    env.start()

    assert env.is_running

    state = env.reset()
    mock_observations.get_latest.assert_called_once()
    mock_action_client.reset_action.assert_called_once()
    assert len(state) == 2
    assert state["cam_0_detected"] == 1
    assert np.all(state["cam_0"] == np.zeros(3))

    assert len(env.curr_state) == 2
    assert env.curr_state["cam_0_detected"] == 1
    assert np.all(env.curr_state["cam_0"] == np.zeros(3))

    # reset mocks
    mock_observations.reset_mock()

    # call step
    raw_action = env.action_space.sample()
    action = RawMovementAction(env.robot_id, raw_action)
    state, _, _, _ = env.step(raw_action)
    mock_observations.get_latest.assert_called_once()
    assert len(mock_action_client.set_action.mock_calls) == 1
    assert (
        mock_action_client.set_action.mock_calls[0].args[0].robot_id == action.robot_id
    )
    assert np.all(
        mock_action_client.set_action.mock_calls[0].args[0].wheel_velocities
        == action.wheel_velocities
    )

    assert len(state) == 2
    assert state["cam_0_detected"] == 1
    assert np.all(state["cam_0"] == np.array([10, 10, np.pi / 2]))

    assert len(env.curr_state) == 2
    assert env.curr_state["cam_0_detected"] == 1
    assert np.all(env.curr_state["cam_0"] == np.array([10, 10, np.pi / 2]))
