import math
from functools import partial

import pytest
import trio

from cmmus_gym.observations.trio import RawVisionObservations
from vision_filter.proto.messages_robocup_ssl_detection_pb2 import \
    SSL_DetectionFrame


def make_default_detection_frame(camera_id: int):
    frame = SSL_DetectionFrame()
    frame.frame_number = 0
    frame.t_capture = 0
    frame.t_sent = 0
    frame.camera_id = camera_id
    return frame


async def start_wrapper(async_fn, *args, task_status=trio.TASK_STATUS_IGNORED):
    task_status.started()
    return await async_fn(*args)


@pytest.fixture
async def detection_channel():
    send, recv = trio.open_memory_channel(math.inf)

    yield send, recv

    await send.aclose()
    await recv.aclose()


async def test_run_with_closed_recv_channel(detection_channel):
    send, recv = detection_channel
    await recv.aclose()

    observations = RawVisionObservations(recv)

    with pytest.raises(trio.ClosedResourceError):
        await observations.run()


async def test_clone_not_running(detection_channel):
    send, recv = detection_channel

    observations = RawVisionObservations(recv)

    detections = await observations.clone()
    assert detections == dict()


async def test_clone_while_running(nursery, detection_channel):
    send, recv = detection_channel

    observations = RawVisionObservations(recv)

    await nursery.start(
        partial(start_wrapper, async_fn=observations.run), name="observations.run",
    )

    # running but nothing has been published to channel
    detections = await observations.clone()
    assert detections == dict()

    cam_0_detection = make_default_detection_frame(0)
    await send.send(cam_0_detection)

    # yield control so that observations can consume channel
    await trio.hazmat.checkpoint()

    detections = await observations.clone()
    assert len(detections) == 1
    assert 0 in detections
    assert detections[0].SerializeToString() == cam_0_detection.SerializeToString()


async def test_clone_clears_old_data(nursery, detection_channel):
    send, recv = detection_channel

    observations = RawVisionObservations(recv)

    await nursery.start(
        partial(start_wrapper, async_fn=observations.run), name="observations.run",
    )

    # running but nothing has been published to channel
    detections = await observations.clone()
    assert detections == dict()

    cam_0_detection = make_default_detection_frame(0)
    await send.send(cam_0_detection)

    # yield control so that observations can consume channel
    await trio.hazmat.checkpoint()

    detections = await observations.clone()
    assert len(detections) == 1
    assert 0 in detections
    assert detections[0].SerializeToString() == cam_0_detection.SerializeToString()

    # should have cleared, and no new data in channel, so detections
    # should be empty
    detections = await observations.clone()
    assert detections == dict()


async def test_clone_multiple_camera_messages_between_updates(
    nursery, detection_channel
):
    send, recv = detection_channel

    observations = RawVisionObservations(recv)

    await nursery.start(
        partial(start_wrapper, async_fn=observations.run), name="observations.run",
    )

    # running but nothing has been published to channel
    detections = await observations.clone()
    assert detections == dict()

    cam_0_detections = [
        make_default_detection_frame(0),
        make_default_detection_frame(0),
        make_default_detection_frame(0),
    ]
    for i, detection in enumerate(cam_0_detections):
        detection.frame_number = i
        await send.send(detection)

    # TEST ONLY: wait until all of the detections have been cleared.
    # In real code you would not want to wait, just grab the latest
    # state as it is available.
    while observations.statistics().num_updates != len(cam_0_detections):
        # yield control so that observations can consume channel
        await trio.hazmat.checkpoint()

    detections = await observations.clone()
    assert len(detections) == 1
    assert 0 in detections
    assert detections[0].SerializeToString() == cam_0_detections[-1].SerializeToString()
