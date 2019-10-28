import struct
from typing import Optional

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
        self,
        multicast_group: str = "224.5.23.2",
        port: int = 10_006,
        detection_messages: Optional[MemorySendChannel] = None,
        geometry_messages: Optional[MemorySendChannel] = None,
    ):
        self._log = structlog.get_logger().bind(
            multicast_group=multicast_group, port=port
        )
        self._log.info("Creating SSL Vision client")
        self.multicast_group = multicast_group
        self.port = port
        self.detection_messages = detection_messages
        self.geometry_messages = geometry_messages

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

    async def recv_message(self, nursery: trio.Nursery):
        self._log.debug("Listening for new message")
        num_bytes = await self._sock.recv_into(self.__recv_view)
        data_view = self.__recv_view[:num_bytes]

        self._log.debug(f"Received {num_bytes} bytes")

        wrapper_packet = SSL_WrapperPacket()
        wrapper_packet.ParseFromString(data_view)

        self._log.debug("Wrapper packet", wrapper_packet=wrapper_packet)
        if wrapper_packet.HasField("detection") and self.detection_messages is not None:
            self._log.debug(
                "Queueing detection message", detection=wrapper_packet.detection
            )
            nursery.start_soon(self.detection_messages.send, wrapper_packet.detection)

        if wrapper_packet.HasField("geometry") and self.geometry_messages is not None:
            self._log.debug(
                "Queueing geometry message", detection=wrapper_packet.geometry
            )
            nursery.start_soon(self.geometry_messages.send, wrapper_packet.geometry)
