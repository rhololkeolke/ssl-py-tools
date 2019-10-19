from threading import Lock

import imgui
import pyglet
import structlog
from imgui.integrations.pyglet import PygletRenderer
from pyglet import gl

from vision_filter.proto.ssl.field.geometry_pb2 import GeometryFieldSize

from .util import Transform


def _make_default_field_geometry() -> GeometryFieldSize:
    field_geometry = GeometryFieldSize()
    field_geometry.field_length = 12000
    field_geometry.field_width = 9000
    field_geometry.goal_width = 1200
    field_geometry.goal_depth = 180
    field_geometry.boundary_width = 250
    return field_geometry


class Visualizer:
    def __init__(self, width: int = 1280, height: int = 720):
        self._log = structlog.get_logger()

        # setup the rendering
        self.window = pyglet.window.Window(width=width, height=height, resizable=True)
        imgui.create_context()
        self.renderer = PygletRenderer(self.window)
        self.window.set_caption("Filter Visualizer")
        self.window.event(self.on_draw)
        self.window.event(self.on_close)

        self.field_texture = None
        self.window.dispatch_event("on_draw")

        # state
        self._field_geometry_lock: Lock = Lock()
        self._field_geometry = _make_default_field_geometry()

        self._set_base_transform()
        self._camera_transform = Transform()

    def _set_base_transform(self):
        right = (
            self._field_geometry.field_length / 2 + self._field_geometry.boundary_width
        )
        left = -right
        top = self._field_geometry.field_width / 2 + self._field_geometry.boundary_width
        bottom = -top

        scaling = min(self.window.width, self.window.height)
        scalex = scaling / (right - left)
        scaley = scaling / (top - bottom)

        if self.window.width < self.window.height:
            transx = -left * scalex
            transy = -bottom * self.window.height / (top - bottom)
        elif self.window.width == self.window.height:
            transx = -left * scalex
            transy = -bottom * scaley
        else:
            transx = -left * self.window.width / (right - left)
            transy = -bottom * scaley

        self._base_transform = Transform(
            translation=(transx, transy), scale=(scalex, scaley)
        )

    def set_field_geometry(self, field_geometry: GeometryFieldSize):
        self._log.debug("set_field_geometry called", field_geometry=field_geometry)
        with self._field_geometry_lock:
            self._field_geometry.CopyFrom(field_geometry)
            self._set_base_transform()

        pyglet.app.platform_event_loop.post_event(self.window, "on_draw")

    def get_field_geometry(self) -> GeometryFieldSize:
        new_field_geometry = GeometryFieldSize()
        with self._field_geometry_lock:
            new_field_geometry.CopyFrom(self._field_geometry)
        return new_field_geometry

    def on_draw(self):
        self._log.debug("on_draw")
        self.draw_field()

        self.draw_imgui()

    def on_close(self):
        """Runs with window.close and if user closes window via OS controls.

        Perform cleanup here. This will shutdown the renderer, and set
        the stored window to None to prevent reruns of window.close().

        Also triggers SIGINT so that other services that have
        registered handlers are cleanedup.

        """
        self._log.debug("on_close")
        self.renderer.shutdown()

    def draw_field(self):
        self._log.debug("draw_field")
        # draw a diagonal white line
        gl.glClearColor(0.133, 0.545, 0.133, 1)
        self.window.clear()

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        field_lines_batch = pyglet.graphics.Batch()
        # TODO(dschwab): Scale/shift this based on camera zoom/position
        with self._field_geometry_lock:
            for line in self._field_geometry.field_lines:
                gl.glLineWidth(line.thickness)
                field_lines_batch.add(
                    2,
                    gl.GL_LINES,
                    None,
                    ("v2d", (line.p1.x, line.p1.y, line.p2.x, line.p2.y)),
                    ("c3B", (255, 255, 255, 255, 255, 255)),
                )

            # TODO(dschwab): Draw arcs

        with self._base_transform:
            with self._camera_transform:
                field_lines_batch.draw()

        self.field_texture = (
            pyglet.image.get_buffer_manager().get_color_buffer().get_texture()
        )

    def draw_imgui(self):
        self._log.debug("draw_imgui")
        self.update_imgui()

        gl.glClearColor(1, 1, 1, 1)
        self.window.clear()
        imgui.render()
        self.renderer.render(imgui.get_draw_data())

    def update_imgui(self):
        self._log.debug("update_imgui")
        imgui.new_frame()
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):
                clicked_load_field_geometry, _ = imgui.menu_item(
                    "Load Field Geometry", "", False, True
                )
                if clicked_load_field_geometry:
                    self._log.warning("Load Field Geometry not implemented")

                clicked_quit, _ = imgui.menu_item("Quit", "Ctrl+Q", False, True)

                if clicked_quit:
                    pyglet.app.platform_event_loop.post_event(self.window, "on_close")

                imgui.end_menu()
            imgui.end_main_menu_bar()

        imgui.show_test_window()

        imgui.begin("Custom window", True)
        imgui.image(
            self.field_texture.id,
            self.field_texture.width,
            self.field_texture.height,
            border_color=(1, 0, 0, 1),
        )
        imgui.end()
