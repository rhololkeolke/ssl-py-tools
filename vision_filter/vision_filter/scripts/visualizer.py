import asyncio
import logging
import os
import signal
from typing import Optional

import click
import pyglet
import structlog
from grpclib.reflection.service import ServerReflection
from grpclib.server import Server
from grpclib.utils import graceful_exit

from vision_filter.ui import Visualizer
from vision_filter.ui.grpc import FilterVisualizerService

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


async def run_pyglet(exit_event: asyncio.Event, framerate: float = 60.0):
    log = structlog.get_logger(__name__)

    log.debug("Starting pyglet run loop")

    while not exit_event.is_set():
        dt = pyglet.clock.tick()
        redraw_all = pyglet.clock.get_default().call_scheduled_functions(dt)
        log.debug("next loop", dt=dt, redraw_all=redraw_all)

        for window in pyglet.app.windows:
            if redraw_all or window.invalid:
                log.debug(
                    "window needs redraw", redraw_all=redraw_all, invalid=window.invalid
                )
                window.switch_to()
                window.dispatch_events()
                window.dispatch_event("on_draw")
                window.flip()

        sleep_time = pyglet.clock.get_sleep_time(True)
        log.debug("Sleeping", sleep_time=sleep_time)
        await asyncio.sleep(sleep_time or 1 / framerate)

    log.debug("Exiting pyglet run loop")


async def signal_grpc_shutdown(exit_event: asyncio.Event, server: Server):
    log = structlog.get_logger(__name__)
    await exit_event.wait()

    log.info("Pyglet shutdown. Triggering GRPC server shutdown.")
    server.close()


async def run_visualizer_server(
    service: FilterVisualizerService,
    exit_event: asyncio.Event,
    host: Optional[str] = None,
    port: int = 50051,
):
    log = structlog.get_logger(__name__, host=host, port=port)
    services = ServerReflection.extend([service])
    server = Server(services)

    signal_grpc_shutdown_task = asyncio.create_task(
        signal_grpc_shutdown(exit_event, server)
    )

    with graceful_exit([server]):
        await server.start(host, port)
        log.info("Started GRPC server")
        await server.wait_closed()
        log.info("Finished GRPC server")
    exit_event.set()

    await asyncio.wait({signal_grpc_shutdown_task})


async def main(host: Optional[str], port: int):
    log = structlog.get_logger(__name__)

    exit_event = asyncio.Event()

    log.info("Creating visualizer")
    visualizer = Visualizer(exit_event)
    pyglet_task = asyncio.create_task(run_pyglet(exit_event))

    log.info("Creating Visualizer GRPC Service")
    service = FilterVisualizerService(visualizer)
    server_task = asyncio.create_task(
        run_visualizer_server(service, exit_event, host, port)
    )

    done, pending = await asyncio.wait({pyglet_task, server_task})

    for task in pending:
        task.cancel()


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

    if log_level:
        log = structlog.get_logger('vision_filter')
        log.setLevel(log_level)

    asyncio.run(main(host, port))
