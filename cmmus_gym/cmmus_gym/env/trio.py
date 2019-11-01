"""Gym environment implementations."""
from typing import Any, Callable, Dict, Tuple

import numpy as np
import structlog
import trio
from gym import spaces
from gym.core import Env

from cmmus_gym.actions.trio import RawMovementAction, RawMovementActionClient
from cmmus_gym.observations.trio import RawVisionObservations
from vision_filter.proto.messages_robocup_ssl_detection_pb2 import \
    SSL_DetectionFrame

from .core import Team


class AsyncEnv(Env):
    """Async compatible version of OpenAI Gym environment."""

    def __init__(self):
        self.__log = structlog.get_logger()

    async def reset(self) -> Any:
        raise NotImplementedError

    async def step(self, action: Any) -> Tuple[Any, Any, bool, Dict[Any, Any]]:
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    async def render(self, mode="human"):
        raise NotImplementedError

    def seed(self, seed: Any = None):
        raise NotImplementedError


_STATE_TYPE = Dict[str, Any]
_ACTION_TYPE = Dict[str, Any]

_REWARD_FUNC_SIG = Callable[[_STATE_TYPE, _ACTION_TYPE, _STATE_TYPE], float]
_IS_TERMINAL_FUNC_SIG = Callable[[_STATE_TYPE, _ACTION_TYPE, _STATE_TYPE], bool]


class SingleRobotMovementEnv(AsyncEnv):
    """Environment with a single robot where tasks involve only movement."""

    def __init__(
        self,
        robot_id: int,
        team: Team,
        num_cameras: int,
        field_width: float,
        field_length: float,
        vision_observations: RawVisionObservations,
        action_client: RawMovementActionClient,
        reward_func: _REWARD_FUNC_SIG,
        is_terminal_func: _IS_TERMINAL_FUNC_SIG,
    ):
        self.__log = structlog.get_logger()
        if robot_id < 0:
            raise ValueError(f"robot_id must be >= 0. Got {robot_id}")
        self._robot_id = robot_id
        if num_cameras < 1:
            raise ValueError(f"num_cameras must be >= 1. Got {num_cameras}")
        self._num_cameras = num_cameras
        self.team = team

        self.field_width = field_width
        self.field_length = field_length

        self.__update_observation_space()

        self.action_space = spaces.Box(low=-1, high=1, shape=(4,))

        self.vision_observations = vision_observations
        self.action_client = action_client

        self._curr_state: _STATE_TYPE = self.__make_empty_observation_state()

        self.reward_func = reward_func
        self.is_terminal_func = is_terminal_func

        self._running = False

    def __update_observation_space(self):
        half_width = self.field_width / 2
        half_length = self.field_length / 2

        # state is [x, y, theta]
        positions = {
            f"cam_{i}": spaces.Box(
                low=np.array([-half_width, -half_length, -np.pi]),
                high=np.array([half_width, half_length, np.pi]),
            )
            for i in range(self.num_cameras)
        }
        detections = {
            f"cam_{i}_detected": spaces.Discrete(2) for i in range(self.num_cameras)
        }

        all_observations = {**positions, **detections}
        self.observation_space = spaces.Dict(all_observations)

    def __make_empty_observation_state(self) -> _STATE_TYPE:
        pass

    @property
    def running(self) -> bool:
        return self._running

    @property
    def robot_id(self) -> int:
        return self._robot_id

    @robot_id.setter
    def robot_id(self, value: int):
        if value < 0:
            raise ValueError(f"robot_id must be >= 0. Got {value}")
        self._robot_id = value

    @property
    def num_cameras(self) -> int:
        return self._num_cameras

    @num_cameras.setter
    def num_cameras(self, value: int):
        if value < 1:
            raise ValueError(f"num_cameras must be >= 1. Got {value}")
        self._num_cameras = value
        self.__update_observation_space()

    @property
    def curr_state(self) -> _STATE_TYPE:
        return self._curr_state

    async def run(self, period: float):
        self._running = True

        async with trio.open_nursery() as nursery:
            nursery.start_soon(self.vision_observations.run)
            nursery.start_soon(self.action_client.run, period)

        self._running = False

    async def reset(self) -> _STATE_TYPE:
        self.__log.debug("reset called", running=self._running)
        if not self._running:
            raise RuntimeError("Cannot call reset until the environment is running.")

        # reset the actions
        await self.action_client.reset_action()

        raw_detections = await self.vision_observations.clone()
        state = self._filter_raw_observations(raw_detections)
        self._curr_state = state
        return state

    async def step(
        self, action: np.ndarray
    ) -> Tuple[_STATE_TYPE, float, bool, Dict[Any, Any]]:
        self.__log.debug("Step called", action=action, running=self._running)
        if not self._running:
            raise RuntimeError("Cannot call step until the environment is running.")

        # convert to proper action type
        raw_movement_action = RawMovementAction(self._robot_id, action.copy())
        await self.action_client.set_action(raw_movement_action)

        raw_detections = await self.vision_observations.clone()
        state = self._filter_raw_observations(raw_detections)

        reward = self.reward_func(self._curr_state, action, state)
        is_terminal = self.is_terminal_func(self._curr_state, action, state)

        self._curr_state = state

        return state, reward, is_terminal, {}

    def close(self):
        raise NotImplementedError

    async def render(self, mode="human"):
        raise NotImplementedError

    def seed(self, seed: Any = None):
        raise NotImplementedError

    def _filter_raw_observations(
        self, raw_detections: Dict[int, SSL_DetectionFrame]
    ) -> _STATE_TYPE:
        state: Dict[str, Any] = {}
        for i in range(self.num_cameras):
            state[f"cam_{i}_detected"] = 0
            state[f"cam_{i}"] = np.zeros(3)
            if i in raw_detections:
                detection = raw_detections[i]
                if self.team == Team.BLUE:
                    for robot_detection in detection.robots_blue:
                        if robot_detection.robot_id == self._robot_id:
                            state[f"cam_{i}_detected"] = 1
                            state[f"cam_{i}"] = np.array(
                                [
                                    robot_detection.x,
                                    robot_detection.y,
                                    robot_detection.orientation,
                                ]
                            )
                            break
                else:
                    for robot_detection in detection.robots_yellow:
                        if robot_detection.robot_id == self._robot_id:
                            state[f"cam_{i}_detected"] = 1
                            state[f"cam_{i}"] = np.array(
                                [
                                    robot_detection.x,
                                    robot_detection.y,
                                    robot_detection.orientation,
                                ]
                            )

        return state
