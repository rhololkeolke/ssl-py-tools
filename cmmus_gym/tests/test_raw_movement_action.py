import mock
import numpy as np
import trio

from cmmus_gym.actions import RawMovementAction, RawMovementActionClient
from cmmus_gym.proto.ssl.radio_pb2 import RobotCommands


class MockCommandStream(mock.MagicMock):
    async def send_message(self, message):
        return self.sync_send_message(message)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return self.sync_aexit(exc_type, exc, tb)


class MockRadioClient(mock.MagicMock):
    def __init__(self):
        super().__init__()

        self._mock_command_stream = MockCommandStream()

        self._command_stream_prop = mock.MagicMock()
        self._command_stream_prop.open.return_value = self._mock_command_stream

    @property
    def CommandStream(self):
        return self._command_stream_prop


async def test_set_action_when_not_running():
    # should not cause any errors
    action_client = RawMovementActionClient(MockRadioClient())
    result = await action_client.set_action(RawMovementAction(0, np.zeros(4)))
    assert result is None


async def test_reset_action_when_not_running():
    # should not cause any errors
    action_client = RawMovementActionClient(MockRadioClient())
    await action_client.reset_action()


async def test_run_sends_at_specified_rate(autojump_clock, nursery):
    radio = MockRadioClient()
    action_client = RawMovementActionClient(radio)

    expected_action = RawMovementAction(0, np.ones(4))
    await action_client.set_action(expected_action)

    expected_sent_message = RobotCommands()
    expected_action.to_proto(expected_sent_message)

    period = 1
    expected_num_sent = 10
    with trio.move_on_after(period * (expected_num_sent - 1)):
        await action_client.run(1)

    mock_command_stream = radio._mock_command_stream
    assert len(mock_command_stream.method_calls) == expected_num_sent
    for call_args in mock_command_stream.sync_send_message.call_args_list:
        assert len(call_args.args) == 1
        assert (
            call_args.args[0].SerializeToString()
            == expected_sent_message.SerializeToString()
        )

    radio = MockRadioClient()
    action_client.radio = radio

    await action_client.reset_action()

    expected_sent_message.Clear()

    with trio.move_on_after(period * (expected_num_sent - 1)):
        await action_client.run(1)

    mock_command_stream = radio._mock_command_stream
    assert len(mock_command_stream.method_calls) == expected_num_sent
    for call_args in mock_command_stream.sync_send_message.call_args_list:
        assert len(call_args.args) == 1
        assert (
            call_args.args[0].SerializeToString()
            == expected_sent_message.SerializeToString()
        )
