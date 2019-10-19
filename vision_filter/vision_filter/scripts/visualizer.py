import logging
from concurrent import futures

import click
import grpc
import pyglet
import structlog
from grpc_reflection.v1alpha import reflection

from vision_filter.proto import (filter_visualizer_pb2,
                                 filter_visualizer_pb2_grpc)
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
    default="[::]",
    type=str,
    help="GRPC host. Default binds to all interfaces.",
)
@click.option("-p", "--port", default=50051, type=int, help="GRPC port.")
def cli(log_level, host, port):
    """Vision Filter Visualizer

    View filtered and unfiltered SSL Vision data. Adjust filter
    parameters and view filter statistics.

    """
    if log_level:
        log = structlog.get_logger("vision_filter")
        log.setLevel(log_level)

    log = structlog.get_logger()

    log.info("Creating visualizer instance")
    visualizer = Visualizer()

    log.info("Creating visualizer grpc server", host=host, port=port)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    filter_visualizer_pb2_grpc.add_FilterVisualizerServicer_to_server(
        FilterVisualizer(visualizer), server
    )
    SERVICE_NAMES = (
        filter_visualizer_pb2.DESCRIPTOR.services_by_name["FilterVisualizer"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()

    log.info("Starting pyglet app loop")
    pyglet.app.run()

    server.stop(None)
