import structlog
from google.protobuf.empty_pb2 import Empty
from grpclib.reflection.service import ServerReflection
from grpclib.server import Server

# generated by protoc
from vision_filter.proto.filter_visualizer_grpc import FilterVisualizerBase
from vision_filter.proto.ssl.field.geometry_pb2 import GeometryFieldSize


class FilterVisualizer(FilterVisualizerBase):
    def __init__(self, host=None, port=50051):
        self._log = structlog.get_logger().bind(host=host, port=port)
        self._server = None

        self.host = host
        self.port = port
        self.field_geometry = GeometryFieldSize()

    def close(self):
        if self._server:
            self._log.info("Closing server")
            self._server.close()
            self._server = None

    async def SetFieldGeometry(self, stream):
        request: GeometryFieldSize = await stream.recv_message()
        self._log.debug("SetFieldGeometry", request=request)
        self.field_geometry = request
        await stream.send_message(Empty())

    async def GetFieldGeometry(self, stream):
        request: Empty = await stream.recv_message()
        self._log.debug(f"GetFieldGeometry", request=request)
        await stream.send_message(self.field_geometry)

    async def run(self):
        services = ServerReflection.extend([self])
        self._server = Server(services)
        await self._server.start(self.host, self.port)
        self._log.info("Started FilterVisualizer GRPC server")
        await self._server.wait_closed()
