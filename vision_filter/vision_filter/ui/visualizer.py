from threading import Lock

import imgui
import pyglet
import structlog
from imgui.integrations.pyglet import PygletRenderer
from pyglet import gl

from vision_filter.proto.ssl.field.geometry_pb2 import GeometryFieldSize


class Visualizer:
    def __init__(self):
        self._log = structlog.get_logger()

        # setup the rendering
        self.window = pyglet.window.Window(width=1280, height=720, resizable=True)
        imgui.create_context()
        self.renderer = PygletRenderer(self.window)
        self.window.set_caption("Filter Visualizer")
        self.window.event(self.on_draw)
        self.window.event(self.on_close)

        self.field_texture = None
        self.window.dispatch_event("on_draw")

        # state
        self._field_geometry_lock: Lock = Lock()
        self._field_geometry: GeometryFieldSize = GeometryFieldSize()

    def set_field_geometry(self, field_geometry: GeometryFieldSize):
        self._log.debug("set_field_geometry called", field_geometry=field_geometry)
        with self._field_geometry_lock:
            self._field_geometry.CopyFrom(field_geometry)

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
        pyglet.gl.glLineWidth(100)

        field_lines_batch = pyglet.graphics.Batch()
        # TODO(dschwab): Scale/shift this based on camera zoom/position
        with self._field_geometry_lock:
            for line in self._field_geometry.field_lines:
                field_lines_batch.add(
                    2,
                    gl.GL_LINES,
                    None,
                    ("v2d", (line.p1.x, line.p1.y, line.p2.x, line.p2.y)),
                    ("c3B", (255, 255, 255, 255, 255, 255)),
                )

            # TODO(dschwab): Draw arcs

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
