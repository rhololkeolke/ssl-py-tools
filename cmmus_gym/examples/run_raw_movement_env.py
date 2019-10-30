#!/usr/bin/env python

import logging
from functools import partial

import click
import numpy as np
import structlog
import trio
import trio_util

from cmmus_gym.actions import (ActionClient, ActionClientStats,
                               RawMovementAction, RawMovementActionClient)
from cmmus_gym.env import SingleRobotMovementEnv, Team
from cmmus_gym.observations import RawVisionObservations
from vision_filter.net import SSLVisionClient

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


async def start_wrapper(async_fn, *args, task_status=trio.TASK_STATUS_IGNORED):
    task_status.started()
    return await async_fn(*args)


class DiscardActionClient(ActionClient):
    def __init__(self):
        self._stats = ActionClientStats()

    async def run(self, period: float):
        pass

    async def set_action(self, action):
        self._stats.last_set_action_time = trio.current_time()

    async def reset_action(self):
        self._stats.last_set_action_time = trio.current_time()

    def statistics(self):
        return self._stats


async def async_main(
    robot_id: int,
    team: Team,
    vision_addr: str,
    vision_port: int,
    radio_addr: str,
    radio_port: int,
    queue_size: int,
):
    log = structlog.get_logger("main").bind(robot_id=robot_id, team=team)
    async with trio.open_nursery() as nursery:
        log.info(
            "Making SSLVisionClient", vision_addr=vision_addr, vision_port=vision_port
        )
        vision_client = await SSLVisionClient.create(vision_addr, vision_port)
        nursery.start_soon(vision_client.run)

        log.info("Creating RawVisionObservations")
        vision_observations = RawVisionObservations(
            await vision_client.create_detection_channel(queue_size)
        )

        log.info("Creating action client")
        action_client = DiscardActionClient()

        log.info("Creating env")
        env = SingleRobotMovementEnv(
            robot_id,
            team,
            4,
            3310,
            5155,
            vision_observations,
            action_client,
            lambda *args: 0.0,
            lambda *args: False,
        )

        log.info("Starting environment")
        await nursery.start(partial(start_wrapper, env.run, 1 / 60))

        while True:
            log.info("New episode")
            curr_state = await env.reset()
            log.info("Initial state", initial_state=curr_state)
            is_terminal = False
            async for _ in trio_util.periodic(1 / 20):
                if is_terminal:
                    break
                action = np.zeros(4)
                log.info("Taking action", action=action)
                state, reward, is_terminal, _ = await env.step(action)
                log.info(
                    "Finished action",
                    last_state=curr_state,
                    state=state,
                    reward=reward,
                    is_terminal=is_terminal,
                )

                curr_state = state

                log.info(
                    "stats",
                    observation_stats=vision_observations.statistics(),
                    action_stats=action_client.statistics(),
                )


@click.command()
@click.argument("robot_id", type=int)
@click.argument("team", type=click.Choice(["blue", "yellow"], case_sensitive=False))
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
@click.option("--queue-size", default=10, type=int, help="Robot detection queue size")
def main(team, log_level, *args, **kwargs):
    """Run SingleRobotMovementEnv with some preset action policies.

    ROBOT_ID The robot that is being controlled
    """
    if team == "blue":
        team = Team.BLUE
    else:
        team = Team.YELLOW

    log = structlog.get_logger("main")
    if log_level:
        log.setLevel(log_level)

    trio.run(partial(async_main, *args, team=team, **kwargs))


if __name__ == "__main__":
    main()
