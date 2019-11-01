"""Asynchronous methods for getting sensor data and other observations."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional

import structlog
import trio

from vision_filter.proto.messages_robocup_ssl_detection_pb2 import \
    SSL_DetectionFrame


@dataclass
class ObservationStats:
    num_updates: int = 0
    last_update_time: Optional[float] = None


class TrioObservation:
    """Stores latest observations obtained asynchronously."""

    async def run(self):
        """Grab updates in a loop.

        Generally should be launched with nursery.start_soon.
        """
        raise NotImplementedError

    async def clone(self):
        """Grab a snapshot of this observation.

        Generally, environments/agents will want whatever is the
        latest data. This method should lock any datastructures, copy
        them and then unlock.

        """
        raise NotImplementedError

    def statistics(self):
        """Useful data about the observations.

        Typically this will include things like how many updates have
        occurred, last time since update, etc.
        """
        raise NotImplementedError


class RawVisionObservations(TrioObservation):
    """Retrieve latest SSL-Vision detections."""

    def __init__(self, detections: trio.MemoryReceiveChannel):
        self._log = structlog.get_logger().bind(detections=detections)

        # data input stream
        self.detections = detections

        # storage for latest state before clone
        self.__detection_lock: trio.Lock = trio.Lock()
        self.__detections: Dict[int, SSL_DetectionFrame] = defaultdict(
            SSL_DetectionFrame
        )

        # statistics
        self.__stats = ObservationStats()

    async def run(self):
        """Grab updates in a loop until channel is closed.

        Generally will be started in parallel to other tasks via
        nursery.start_soon.

        See Also
        --------
        update
        """
        async with self.detections:
            async for detection in self.detections:
                async with self.__detection_lock:
                    self._log.debug("New frame", camera_id=detection.camera_id)
                    frame = self.__detections[detection.camera_id]
                    frame.CopyFrom(detection)
                self.__stats.num_updates += 1
                self.__stats.last_update_time = trio.current_time()

    async def clone(self) -> Dict[int, SSL_DetectionFrame]:
        """Clone and clear latest camera detections."""
        async with self.__detection_lock:
            self._log.debug("Cloning observations", detections=self.__detections)
            detections: Dict[int, SSL_DetectionFrame] = dict()
            for cam_id, detection_frame in self.__detections.items():
                detections[cam_id] = SSL_DetectionFrame()
                detections[cam_id].CopyFrom(detection_frame)

            # clear so we don't accidentally keep detections from
            # previous observations if we didn't receive an update on
            # that camera since last clone
            self.__detections.clear()
        return detections

    def statistics(self) -> ObservationStats:
        """Returns the current observation stats."""
        return self.__stats
