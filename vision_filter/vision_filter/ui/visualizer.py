import asyncio
import logging
import os
import signal

import imgui
import pyglet
import structlog
from imgui.integrations.pyglet import PygletRenderer
from pyglet import gl


def _run_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


class Visualizer:
    def __init__(self):
        self._log = structlog.get_logger()
        self._log.setLevel(logging.INFO)

    def run(self):
        self._log.debug("Running visualizer")

        # setup the rendering
        self.window = pyglet.window.Window(width=1280, height=720, resizable=True)
        imgui.create_context()
        self.renderer = PygletRenderer(self.window)
        self.window.set_caption("Filter Visualizer")
        self.window.event(self.on_draw)
        self.window.event(self.on_close)

        self.field_texture = None
        self.window.dispatch_event("on_draw")

        pyglet.app.run()

    def exit(self):
        # close the window if not already closed
        if self.window:
            self.window.close()

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
        self.window = None

        os.kill(os.getpid(), signal.SIGINT)

    def draw_field(self):
        self._log.debug("draw_field")
        # draw a diagonal white line
        gl.glClearColor(0.133, 0.545, 0.133, 1)
        self.window.clear()
        pyglet.gl.glLineWidth(100)
        pyglet.graphics.draw(
            2,
            gl.GL_LINES,
            ("v2i", (-10000, -10000, 10000, 10000)),
            ("c3B", (255, 255, 255, 255, 255, 255)),
        )

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
                clicked_quit, selected_quit = imgui.menu_item(
                    "Quit", "Ctrl+Q", False, True
                )

                if clicked_quit:
                    self.exit()

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
