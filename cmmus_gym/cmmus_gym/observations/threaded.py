import socket
import struct
import time
from collections import defaultdict
from copy import deepcopy
from threading import Event, Lock, Thread
from typing import Any, Dict

import structlog
from google.protobuf.message import DecodeError

from vision_filter.proto.messages_robocup_ssl_detection_pb2 import \
    SSL_DetectionFrame
from vision_filter.proto.messages_robocup_ssl_wrapper_pb2 import \
    SSL_WrapperPacket

from .core import ObservationStats


class ThreadedObservation(Thread):
    def __init__(self):
        super().__init__()
        self._stats_lock = Lock()
        self._stats = ObservationStats()

        self._stop_event = Event()

    def run(self):
        raise NotImplementedError

    def get_latest(self) -> Any:
        raise NotImplementedError

    def statistics(self) -> ObservationStats:
        with self._stats_lock:
            return deepcopy(self._stats)

    def stop(self):
        self._stop_event.set()


class RawVisionObservations(ThreadedObservation):
    def __init__(self, multicast_group: str = "224.5.23.2", port: int = 10_006):
        super().__init__()

        # logging setup
        self._log = structlog.get_logger().bind(
            multicast_group=multicast_group, port=port
        )
        self._log.debug("Creating RawVisionObservations")

        # properties
        self._multicast_group = multicast_group
        self._port = port

        # create the multicast receiver socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.bind((self._multicast_group, self._port))
        mreq = struct.pack(
            "4sl", socket.inet_aton(self._multicast_group), socket.INADDR_ANY
        )
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        self._sock = sock

        # don't create a new bytestring everytime we receive a UDP
        # packet
        self.__recv_buf = bytearray(b" " * 8192)
        self.__recv_view = memoryview(self.__recv_buf)

        # state and threading synchronization
        self.__detections_lock = Lock()
        self.__detections: Dict[int, SSL_DetectionFrame] = defaultdict(
            SSL_DetectionFrame
        )

    @property
    def multicast_group(self) -> str:
        return self._multicast_group

    @property
    def port(self) -> int:
        return self._port

    def run(self):
        self._log.debug("Running raw vision observations thread")
        self._stop_event.clear()

        while not self._stop_event.is_set():
            num_bytes = self._sock.recv_into(self.__recv_view)
            data_view = self.__recv_view[:num_bytes]

            self._log.debug(f"Received {num_bytes} bytes")

            try:
                wrapper_packet = SSL_WrapperPacket()
                wrapper_packet.ParseFromString(data_view)
            except DecodeError as e:
                self._log.error(
                    "Failed to decode wrapper packet",
                    raw_data=data_view.tobytes(),
                    error=e,
                )
                continue

            self._log.debug("Wrapper packet", wrapper_packet=wrapper_packet)

            with self._stats_lock:
                self._stats.num_updates += 1
                self._stats.last_update_time = time.time()
                self._stats.last_update_time_ns = time.time_ns()

            if wrapper_packet.HasField("detection"):
                with self.__detections_lock:
                    camera_id = wrapper_packet.detection.camera_id
                    self._log.debug("New frame", camera_id=camera_id)
                    frame = self.__detections[camera_id]
                    frame.CopyFrom(wrapper_packet.detection)

    def get_latest(self) -> Dict[int, SSL_DetectionFrame]:
        with self.__detections_lock:
            self._log.debug("Copying observations", detections=self.__detections)
            detections: Dict[int, SSL_DetectionFrame] = {}
            for cam_id, detection_frame in self.__detections.items():
                # Don't need to do CopyFrom because we're going to
                # clear the internal detections dict
                detections[cam_id] = detection_frame

            # don't want to keep observations from previous state, so
            # clear the internal state whenever this method is called
            self.__detections.clear()

        return detections
