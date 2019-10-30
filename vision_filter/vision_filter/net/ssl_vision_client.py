import math
import struct
from typing import List, Literal, Optional, Union

import structlog
import trio
from trio import MemorySendChannel, socket

from vision_filter.proto.messages_robocup_ssl_wrapper_pb2 import \
    SSL_WrapperPacket


class SSLVisionClient:
    @classmethod
    async def create(cls, *args, **kwargs):
        client = SSLVisionClient(*args, **kwargs)
        await client._create_socket()
        return client

    def __init__(
        self, multicast_group: str = "224.5.23.2", port: int = 10_006,
    ):
        self._log = structlog.get_logger().bind(
            multicast_group=multicast_group, port=port
        )
        self._log.info("Creating SSL Vision client")
        self.multicast_group = multicast_group
        self.port = port

        self._detection_channels: List[MemorySendChannel] = []
        self._geometry_channels: List[MemorySendChannel] = []

        # don't create a new bytestring everytime we receive a UDP
        # packet
        self.__recv_buf = bytearray(b" " * 8192)
        self.__recv_view = memoryview(self.__recv_buf)

    async def _create_socket(self):
        self._sock = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
        )
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        await self._sock.bind((self.multicast_group, self.port))
        mreq = struct.pack(
            "4sl", socket.inet_aton(self.multicast_group), socket.INADDR_ANY
        )
        self._sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    def create_detection_channel(
        self, queue_size: Union[int, Literal[math.inf]]  # type: ignore
    ):
        send, recv = trio.open_memory_channel(queue_size)
        self._detection_channels.append(send)
        return recv

    def create_geometry_channel(
        self, queue_size: Union[int, Literal[math.inf]]  # type: ignore
    ):
        send, recv = trio.open_memory_channel(queue_size)
        self._geometry_channels.append(send)
        return recv

    async def recv_message(self, nursery: trio.Nursery):
        self._log.debug("Listening for new message")
        num_bytes = await self._sock.recv_into(self.__recv_view)
        data_view = self.__recv_view[:num_bytes]

        self._log.debug(f"Received {num_bytes} bytes")

        wrapper_packet = SSL_WrapperPacket()
        wrapper_packet.ParseFromString(data_view)

        self._log.debug("Wrapper packet", wrapper_packet=wrapper_packet)
        if wrapper_packet.HasField("detection"):
            self._log.debug(
                "Queueing detection message", detection=wrapper_packet.detection
            )
            for chan in self._detection_channels:
                nursery.start_soon(chan.send, wrapper_packet.detection)

        if wrapper_packet.HasField("geometry"):
            self._log.debug(
                "Queueing geometry message", detection=wrapper_packet.geometry
            )
            for chan in self._geometry_channels:
                nursery.start_soon(chan.send, wrapper_packet.geometry)
