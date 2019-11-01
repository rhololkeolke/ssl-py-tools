#!/usr/bin/env python

import contextlib
import logging
import time

import click
import grpc
import numpy as np
import structlog

from cmmus_gym.actions import RawMovementAction
from cmmus_gym.actions.threaded import RawMovementActionClient
from cmmus_gym.proto.ssl.radio_pb2_grpc import RadioStub

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


@contextlib.contextmanager
def run_action_client(radio: RadioStub, action_period: float):
    action_client = RawMovementActionClient(radio, action_period)
    action_client.start()

    try:
        yield action_client
    finally:
        action_client.stop()
        action_client.join()


@click.command()
@click.argument("robot_id", type=int)
@click.option(
    "-l",
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    envvar="LOG_LEVEL",
    help="Set the log level",
)
@click.option(
    "--radio-addr", default="127.0.0.1", type=str, help="Radio server address"
)
@click.option("--radio-port", default=50_051, type=int, help="Radio server port")
def main(robot_id, log_level, radio_addr, radio_port):
    """Run SingleRobotMovementActionClient with preset commands.

    ROBOT_ID The robot that is being controlled
    """
    log = structlog.get_logger("main")
    if log_level:
        log.setLevel(log_level)

    log.info(f"Creating action client channel", addr=radio_addr, port=radio_port)
    with grpc.insecure_channel(f"{radio_addr}:{radio_port}") as channel:
        radio = RadioStub(channel)
        with run_action_client(radio, 1 / 60) as action_client:
            action_client.set_action(RawMovementAction(robot_id, 0.01 * np.ones(4)))
            while True:
                log.info("Latest stats", stats=action_client.statistics())
                time.sleep(1)


if __name__ == "__main__":
    main()
