import ctypes
import math
from threading import Lock

import imgui
import pyglet
import structlog
from imgui.integrations.pyglet import PygletRenderer
from pyglet import gl

from vision_filter.proto.ssl.field.geometry_pb2 import GeometryFieldSize

from .field_geometry_editor import FieldGeometryEditor
from .util import Transform


def _make_default_field_geometry() -> GeometryFieldSize:
    field_geometry = GeometryFieldSize()
    field_geometry.field_length = 12000
    field_geometry.field_width = 9000
    field_geometry.goal_width = 1200
    field_geometry.goal_depth = 180
    field_geometry.boundary_width = 250
    return field_geometry


class LineGroup(pyglet.graphics.Group):
    def __init__(self, line_width: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.line_width = line_width

    def set_state(self):
        gl.glLineWidth(self.line_width)

    def unset_state(self):
        gl.glLineWidth(1.0)


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

        # create field framebuffer
        self.field_buf = gl.GLuint(0)
        gl.glGenFramebuffers(1, ctypes.byref(self.field_buf))
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.field_buf)

        # create field texture
        self.field_texture = gl.GLuint(0)
        self.field_texture_width = width
        self.field_texture_height = height
        gl.glGenTextures(1, ctypes.byref(self.field_texture))
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.field_texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA,
            width,
            height,
            0,
            gl.GL_RGBA,
            gl.GL_FLOAT,
            None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        # bind field texture to field framebuffer
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER,
            gl.GL_COLOR_ATTACHMENT0,
            gl.GL_TEXTURE_2D,
            self.field_texture,
            0,
        )

        # Something may have gone wrong during the process, depending on the
        # capabilities of the GPU.
        res = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if res != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Framebuffer not completed")

        self.window.dispatch_event("on_draw")

        # state
        self._field_geometry_lock: Lock = Lock()
        self._field_geometry = _make_default_field_geometry()

        self._set_base_transform()
        self._camera_transform = Transform()

        self._field_geometry_editor = FieldGeometryEditor(False)

    def _set_base_transform(self):
        right = (
            self._field_geometry.field_length / 2 + self._field_geometry.boundary_width
        )
        left = -right
        top = self._field_geometry.field_width / 2 + self._field_geometry.boundary_width
        bottom = -top

        field_aspect_ratio = (right - left) / (top - bottom)

        scaley = self.field_texture_height / (top - bottom)
        scalex = self.field_texture_width * field_aspect_ratio / (right - left)

        transx = -left * self.window.width / (right - left)
        transy = -bottom * self.window.height / (top - bottom)

        self._base_transform = Transform(
            translation=(transx, transy), scale=(scalex, scaley)
        )

    def on_close(self):
        """Runs with window.close and if user closes window via OS controls.

        Perform cleanup here. This will shutdown the renderer, and set
        the stored window to None to prevent reruns of window.close().

        Also triggers SIGINT so that other services that have
        registered handlers are cleanedup.

        """
        self._log.debug("on_close")
        self.renderer.shutdown()

    def set_field_geometry(self, field_geometry: GeometryFieldSize):
        self._log.debug("set_field_geometry called", field_geometry=field_geometry)
        with self._field_geometry_lock:
            self._field_geometry.CopyFrom(field_geometry)
            self._field_line_expanded = [False] * len(self._field_geometry.field_lines)
            self._field_arc_expanded = [False] * len(self._field_geometry.field_arc)
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

    def draw_field(self):
        self._log.debug("draw_field")
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.field_buf)
        gl.glViewport(0, 0, self.field_texture_width, self.field_texture_height)
        gl.glClearColor(0.133, 0.545, 0.133, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        field_lines_batch = pyglet.graphics.Batch()
        # TODO(dschwab): Scale/shift this based on camera zoom/position
        with self._field_geometry_lock:
            for line in self._field_geometry.field_lines:
                field_lines_batch.add(
                    2,
                    gl.GL_LINES,
                    LineGroup(line.thickness / 100),
                    ("v2d", (line.p1.x, line.p1.y, line.p2.x, line.p2.y)),
                    ("c3B", (255, 255, 255, 255, 255, 255)),
                )

            for arc in self._field_geometry.field_arcs:
                points = []
                res = 30
                for i in range(res):
                    ang = arc.a1 + i * (arc.a2 - arc.a1) / res
                    points.extend(
                        [
                            math.cos(ang) * arc.radius + arc.center.x,
                            math.sin(ang) * arc.radius + arc.center.y,
                        ]
                    )
                field_lines_batch.add(
                    res,
                    gl.GL_LINE_LOOP,
                    LineGroup(arc.thickness / 100),
                    ("v2d", points),
                    ("c3B", [255 for _ in range(3 * res)]),
                )

        with self._base_transform:
            with self._camera_transform:
                field_lines_batch.draw()

    def draw_imgui(self):
        self._log.debug("draw_imgui")
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glViewport(0, 0, self.window.width, self.window.height)

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
            if imgui.begin_menu("Edit", True):
                clicked_edit_field_geometry, _ = imgui.menu_item(
                    "Edit Field Geometry", "", False, True
                )
                if clicked_edit_field_geometry:
                    self._field_geometry_editor.visible = True

                imgui.end_menu()
            imgui.end_main_menu_bar()

        # imgui.show_test_window()

        imgui.set_next_window_position(0, 0)
        display_size = imgui.get_io().display_size
        imgui.set_next_window_size(display_size.x, display_size.y)
        imgui.begin(
            "Field",
            flags=imgui.WINDOW_NO_TITLE_BAR
            | imgui.WINDOW_NO_SCROLLBAR
            | imgui.WINDOW_MENU_BAR
            | imgui.WINDOW_NO_MOVE
            | imgui.WINDOW_NO_RESIZE
            | imgui.WINDOW_NO_COLLAPSE
            | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS,
        )

        region_available = imgui.get_content_region_available()

        image_aspect_ratio = self.field_texture_width / self.field_texture_height
        height = region_available.y
        width = height * image_aspect_ratio

        cursor_pos = imgui.get_cursor_pos()
        cursor_x = (
            max(0, (region_available.x - width - cursor_pos.x) / 2) + cursor_pos.x
        )
        cursor_y = (
            max(0, (region_available.y - height - cursor_pos.y) / 2) + cursor_pos.y
        )
        imgui.set_cursor_pos((cursor_x, cursor_y))
        imgui.image(self.field_texture, width, height)

        imgui.end()

        if self._field_geometry_editor.visible:
            with self._field_geometry_lock:
                self._field_geometry_editor(self._field_geometry)
