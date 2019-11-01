import contextlib
from functools import partial
from typing import List

import mock
from hypothesis import given, settings
from hypothesis.provisional import ip4_addr_strings
from hypothesis.strategies import integers, lists

from cmmus_gym.observations.threaded import RawVisionObservations
from vision_filter.proto.messages_robocup_ssl_wrapper_pb2 import \
    SSL_WrapperPacket


def mock_socket_recv_into(buf: bytearray, data: bytearray):
    buf[: len(data)] = data
    return len(data)


class RecvIntoSeq:
    def __init__(self, datas: List[bytearray]):
        self.datas = datas
        self.index = 0

    def __call__(self, buf: bytearray):
        if self.index >= len(self.datas):
            return mock_socket_recv_into(buf, bytearray(0))
        else:
            return_value = mock_socket_recv_into(buf, self.datas[self.index])
            self.index += 1
            return return_value


def make_observations(*args, **kwargs):
    with mock.patch("socket.socket"):
        observations = RawVisionObservations(*args, **kwargs)
        # by default return no bytes
        observations._sock.recv_into.side_effect = partial(
            mock_socket_recv_into, data=bytearray(0)
        )
        return observations


@contextlib.contextmanager
def launch_observations(observations):
    try:
        observations.start()
        yield observations
    finally:
        observations.stop()
        observations.join()


@given(multicast_group=ip4_addr_strings(), port=integers(min_value=1, max_value=65_535))
def test_properties(multicast_group: str, port: int):
    observations = make_observations(multicast_group, port)

    assert observations.multicast_group == multicast_group
    assert observations.port == port


def test_get_latest_not_running():
    observations = make_observations()

    assert not observations.is_alive()

    detections = observations.get_latest()
    assert detections == {}

    stats = observations.statistics()
    assert stats.num_updates == 0
    assert stats.last_update_time is None
    assert stats.last_update_time_ns is None


def test_bad_decode():
    observations = make_observations()
    observations._sock.recv_into.side_effect = partial(
        mock_socket_recv_into, data=bytearray(b"bad")
    )

    with launch_observations(observations):
        assert observations.is_alive()

        # wait until we've read from socket at least once
        while len(observations._sock.recv_into.mock_calls) == 0:
            pass

        # verify that a bad decode doesn't crash thread
        assert observations.is_alive()


def test_get_latest_while_running():
    observations = make_observations()

    with launch_observations(observations):
        assert observations.is_alive()

        # wait until we've read from socket at least once
        while len(observations._sock.recv_into.mock_calls) == 0:
            pass

    # we didn't put any data into the mock socket, so we should
    # get an empty dict
    detections = observations.get_latest()
    assert detections == {}

    stats = observations.statistics()
    assert stats.num_updates == len(observations._sock.recv_into.mock_calls)
    assert stats.last_update_time is not None
    assert stats.last_update_time_ns is not None


@given(
    cam_ids=lists(
        integers(min_value=0, max_value=10), min_size=1, max_size=5, unique=True
    )
)
# FIXME(dschwab): Not sure why but larger numbers cause a hypothesis
# deadline timeout. I was not expecting this test to take so long.
@settings(max_examples=1)
def test_get_latest_clears_state(cam_ids):
    # generate fake cam detection messages
    expected_detections = {}
    data_sequence = []
    for frame_number, cam_id in enumerate(cam_ids):
        wrapper = SSL_WrapperPacket()
        wrapper.detection.camera_id = cam_id
        wrapper.detection.t_capture = float(frame_number)
        wrapper.detection.t_sent = float(frame_number)
        wrapper.detection.frame_number = frame_number
        data_sequence.append(wrapper.SerializeToString())
        expected_detections[cam_id] = wrapper.detection

    observations = make_observations()
    observations._sock.recv_into.side_effect = RecvIntoSeq(data_sequence)

    with launch_observations(observations):
        assert observations.is_alive()

        # wait until all of the messages have been received and decoded
        while len(observations._sock.recv_into.mock_calls) < len(data_sequence):
            pass

    # make sure we got all the messages
    detections = observations.get_latest()
    assert detections == expected_detections

    # now the list should be cleared
    detections = observations.get_latest()
    assert detections == {}

    stats = observations.statistics()
    assert stats.num_updates >= len(expected_detections)


@given(
    cam_id=integers(min_value=0, max_value=10),
    num_duplicates=integers(min_value=1, max_value=5),
)
# FIXME(dschwab): Not sure why but larger numbers cause a hypothesis
# deadline timeout. I was not expecting this test to take so long.
@settings(max_examples=5)
def test_get_latest_duplicates_overwritten(cam_id, num_duplicates):
    # generate fake cam detection messages
    expected_detections = {}
    data_sequence = []
    for frame_number in range(num_duplicates):
        wrapper = SSL_WrapperPacket()
        wrapper.detection.camera_id = cam_id
        wrapper.detection.t_capture = float(frame_number)
        wrapper.detection.t_sent = float(frame_number)
        wrapper.detection.frame_number = frame_number
        data_sequence.append(wrapper.SerializeToString())
        expected_detections[cam_id] = wrapper.detection

    observations = make_observations()
    observations._sock.recv_into.side_effect = RecvIntoSeq(data_sequence)

    with launch_observations(observations):
        assert observations.is_alive()

        # wait until all of the messages have been received and decoded
        while len(observations._sock.recv_into.mock_calls) < len(data_sequence):
            pass

    # make sure we got all the messages
    detections = observations.get_latest()
    assert detections == expected_detections

    stats = observations.statistics()
    assert stats.num_updates >= num_duplicates
