import logging
import math
from typing import Literal, Optional, Union

import click
import structlog
import trio

from vision_filter.net import SSLVisionClient, SSLVisionServer

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


async def print_messages(recv_channel: trio.MemoryReceiveChannel):
    async with recv_channel:
        async for message in recv_channel:
            print(message)


async def main(
    log: logging.Logger,
    multicast_group: str,
    recv_port: int,
    send_port: int,
    queue_size: Union[int, Literal[math.inf]],  # type: ignore
):
    log.info("Using message queue size of {queue_size}")

    log.info(
        "Creating SSL Vision Client",
        multicast_group=multicast_group,
        recv_port=recv_port,
    )
    vision_client = await SSLVisionClient.create(multicast_group, recv_port)

    print_detection_recv = await vision_client.create_detection_channel(queue_size)
    print_geometry_recv = await vision_client.create_geometry_channel(queue_size)

    repub_detection_recv = await vision_client.create_detection_channel(queue_size)
    repub_geometry_recv = await vision_client.create_geometry_channel(queue_size)

    log.info(
        "Creating SSL Vision Server",
        multicast_group=multicast_group,
        send_port=send_port,
    )
    vision_server = SSLVisionServer(
        repub_detection_recv, repub_geometry_recv, multicast_group, send_port
    )

    async with trio.open_nursery() as nursery:
        nursery.start_soon(vision_server.send_messages)
        nursery.start_soon(print_messages, print_detection_recv)
        nursery.start_soon(print_messages, print_geometry_recv)
        while True:
            await vision_client.recv_message(nursery)


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
    "--multicast-group", default="224.5.23.2", type=str, help="Multicast group."
)
@click.option("--recv-port", default=10006, type=int, help="Receiver port.")
@click.option("--send-port", default=10007, type=int, help="Retransmit port.")
@click.option(
    "--queue-size",
    default=None,
    type=int,
    help="Maximum message queue size before blocking. Defaults to infinity.",
)
def cli(
    log_level: str,
    multicast_group: str,
    recv_port: int,
    send_port: int,
    queue_size: Optional[int],
):
    """Receive ssl-vision packets, filter and republish."""
    log = structlog.get_logger()
    if log_level:
        log.setLevel(log_level)

    trio.run(main, log, multicast_group, recv_port, send_port, queue_size or math.inf)
