#!/usr/bin/env python

import contextlib
import logging
import time

import click
import structlog

from cmmus_gym.observations.threaded import RawVisionObservations

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
def run_observations(multicast_group: str, port: int):
    observations = RawVisionObservations(multicast_group, port)
    observations.start()

    try:
        yield observations
    finally:
        observations.stop()
        observations.join()


@click.command()
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
def main(log_level, vision_addr, vision_port):
    log = structlog.get_logger("main")
    if log_level:
        log.setLevel(log_level)

    log.info(f"Creating observations for {vision_addr}:{vision_port}")
    with run_observations(vision_addr, vision_port) as observations:
        while True:
            log.info("Latest observations", detections=observations.get_latest())
            log.info("Latest stats", stats=observations.statistics())
            time.sleep(1)


if __name__ == "__main__":
    main()
