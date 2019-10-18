import asyncio
import logging
import signal
from threading import Thread

import click
import structlog

from vision_filter.ui import Visualizer
from vision_filter.ui.grpc import FilterVisualizer

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


@click.command()
@click.option(
    "-l",
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    envvar="LOG_LEVEL",
    help="Path to non-default logging configuration.",
)
@click.option(
    "-h",
    "--host",
    default=None,
    type=str,
    help="GRPC host. Default binds to all interfaces.",
)
@click.option("-p", "--port", default=50051, type=int, help="GRPC port.")
def cli(log_level, host, port):
    """Vision Filter Visualizer

    View filtered and unfiltered SSL Vision data. Adjust filter
    parameters and view filter statistics.

    """
    log = structlog.get_logger("visualizer")
    if log_level:
        logging.getLogger().setLevel(log_level)

    log.info("Starting Visualizer")
    visualizer = Visualizer()
    signal.signal(signal.SIGINT, lambda sig, frame: visualizer.exit())
    visualizer_thread = Thread(target=visualizer.run)
    visualizer_thread.start()

    log.info("Starting Visualizer GRPC Server")
    server = FilterVisualizer(visualizer, host=host, port=port)
    asyncio.run(server.run())
