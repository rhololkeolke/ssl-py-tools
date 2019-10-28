import socket
from typing import Callable

import structlog
import trio
from google.protobuf.message import Message
from trio import MemoryReceiveChannel

from vision_filter.proto.messages_robocup_ssl_wrapper_pb2 import \
    SSL_WrapperPacket


class SSLVisionServer:
    def __init__(
        self,
        detection_messages: MemoryReceiveChannel,
        geometry_messages: MemoryReceiveChannel,
        multicast_group: str = "224.5.23.2",
        port: int = 10_006,
        multicast_ttl: int = 32,
    ):
        self._log = structlog.get_logger().bind(
            multicast_group=multicast_group, port=port
        )
        self._log.setLevel("DEBUG")
        self._log.info("Creating SSL Vision client")
        self.multicast_group = multicast_group
        self.port = port
        self.detection_messages = detection_messages
        self.geometry_messages = geometry_messages

        self.__create_socket(multicast_ttl)

    def __create_socket(self, multicast_ttl: int):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, multicast_ttl)
        self._sock = trio.socket.from_stdlib_socket(sock)

    async def send_messages(self):
        def construct_detection_wrapper(wrapper: SSL_WrapperPacket, detection: Message):
            wrapper.detection.CopyFrom(detection)

        def construct_geometry_wrapper(wrapper: SSL_WrapperPacket, geometry: Message):
            wrapper.geometry.CopyFrom(geometry)

        async with trio.open_nursery() as nursery:
            nursery.start_soon(
                self._send_messages,
                self.detection_messages,
                construct_detection_wrapper,
            )
            nursery.start_soon(
                self._send_messages, self.geometry_messages, construct_geometry_wrapper
            )

    async def _send_messages(
        self,
        channel: MemoryReceiveChannel,
        construct_wrapper: Callable[[SSL_WrapperPacket, Message], None],
    ):
        wrapper = SSL_WrapperPacket()
        async with channel:
            async for message in channel:
                construct_wrapper(wrapper, message)
                self._log.debug("Sending new wrapper message", wrapper=wrapper)
                await self._sock.sendto(
                    wrapper.SerializeToString(), (self.multicast_group, self.port)
                )
