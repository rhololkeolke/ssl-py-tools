#!/usr/bin/env python

import contextlib
import logging
import time
from typing import Optional

import click
import grpc
import numpy as np
import structlog
from gym.core import Env

from cmmus_gym.actions.threaded import RawMovementActionClient
from cmmus_gym.env import Team
from cmmus_gym.env.threaded import SingleRobotRawMovementEnv
from cmmus_gym.observations.threaded import RawVisionObservations
from cmmus_gym.proto.ssl.radio_pb2_grpc import RadioStub

NUM_CAMERAS = 4
FIELD_LENGTH = 5155
FIELD_WIDTH = 3310

logging.basicConfig(format="%(message)s")
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)


def zero_reward(*args):
    return 0.0


def never_terminate(*args):
    return False


@contextlib.contextmanager
def manage_environment(env):
    try:
        env.start()
        yield env
    finally:
        env.close()


def run_environment(
    log,
    env: Env,
    env_freq: float,
    max_episode_steps: Optional[int] = None,
    max_steps: Optional[int] = None,
    max_episodes: Optional[int] = None,
):
    log = log.bind(
        max_episode_steps=max_episode_steps,
        max_steps=max_steps,
        max_episodes=max_episodes,
        env_freq=env_freq,
    )

    num_episodes = 0
    num_steps = 0
    while (max_episodes is None or num_episodes < max_episodes) and (
        max_steps is None or num_steps < max_steps
    ):
        log.info("Starting new episode", num_episodes=num_episodes, num_steps=num_steps)
        num_episode_steps = 0

        is_terminal = False
        curr_state = env.reset()
        while (max_steps is None or num_steps < max_steps) and (
            max_episode_steps is None or num_episode_steps < max_episode_steps
        ):
            start_time = time.time()

            if is_terminal:
                break

            action = 0.3 * np.random.rand(4)
            log.info("Taking action", action=action)
            state, reward, is_terminal, debug = env.step(action)
            log.info(
                "Finished step",
                last_state=curr_state,
                state=state,
                reward=reward,
                is_terminal=is_terminal,
            )
            curr_state = state

            log.info("debug info", **debug)

            elapsed_time = time.time() - start_time

            log.debug(f"elapsed_time={elapsed_time}")


@click.command()
@click.argument("robot_id", type=int)
@click.argument("team", type=click.Choice(["blue", "yellow"], case_sensitive=False))
@click.argument("env_freq", type=float)
@click.option(
    "-l",
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    envvar="LOG_LEVEL",
    help="Set the log level",
)
@click.option(
    "--vision-addr",
    default="224.5.23.2",
    type=str,
    help="SSL-Vision multicast group address.",
)
@click.option(
    "--vision-port", default=10_006, type=int, help="SSL-Vision multicast port"
)
@click.option(
    "--radio-addr", default="127.0.0.1", type=str, help="Radio server address"
)
@click.option("--radio-port", default=50_051, type=int, help="Radio server port")
@click.option(
    "--max-episode-steps",
    default=None,
    type=int,
    help="Maximum number of steps allowed in an episode",
)
@click.option(
    "--max-steps",
    default=None,
    type=int,
    help="Maximum number of steps across all episodes",
)
@click.option(
    "--max-episodes", default=None, type=int, help="Maximum number of episodes"
)
def main(
    robot_id,
    team,
    env_freq,
    log_level,
    vision_addr,
    vision_port,
    radio_addr,
    radio_port,
    max_episode_steps,
    max_steps,
    max_episodes,
):
    """Run SingleRobotRawMovementEnv with some preset action policies.

    ROBOT_ID The robot being controlled
    TEAM     The team of the controlled robot
    ENV_FREQ How often a new action should be set
    """
    if team == "blue":
        team = Team.BLUE
    else:
        team = Team.YELLOW

    log = structlog.get_logger("main")
    if log_level:
        log.setLevel(log_level)

    observations = RawVisionObservations(vision_addr, vision_port)
    with grpc.insecure_channel(f"{radio_addr}:{radio_port}") as channel:
        action_client = RawMovementActionClient(RadioStub(channel), 1 / 60)

        env = SingleRobotRawMovementEnv(
            robot_id,
            team,
            4,
            FIELD_WIDTH,
            FIELD_LENGTH,
            observations,
            action_client,
            zero_reward,
            never_terminate,
            env_freq,
        )
        with manage_environment(env):
            run_environment(
                log, env, env_freq, max_episode_steps, max_steps, max_episodes
            )


if __name__ == "__main__":
    main()
