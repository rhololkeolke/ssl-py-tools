import asyncio
import logging
from threading import Thread

import imgui
import pyglet
import structlog
from imgui.integrations.pyglet import PygletRenderer
from pyglet import gl

from .grpc import FilterVisualizer


def _run_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


class Visualizer:
    def __init__(self, host=None, port=50051):
        self._log = structlog.get_logger()
        self._log.setLevel(logging.INFO)

        # setup the rendering
        self.window = pyglet.window.Window(width=1280, height=720, resizable=True)
        imgui.create_context()
        self.renderer = PygletRenderer(self.window)
        self.window.event(self.on_draw)

        self.field_texture = None
        self.on_draw()  # force first draw so we don't get a blank window

        # setup the server
        self._server_loop = asyncio.new_event_loop()
        Thread(target=lambda: _run_loop(self._server_loop)).start()
        self._server = FilterVisualizer(host=host, port=port)
        self._server_future = asyncio.run_coroutine_threadsafe(
            self._server.run(), self._server_loop
        )

    def run(self):
        self._log.info("Run")
        pyglet.app.run()

    def exit(self):
        self._log.info("Exit called")
        self.renderer.shutdown()
        self._server.close()
        self._server_loop.call_soon_threadsafe(self._server_loop.stop)
        try:
            self._server_future.result(1)
        except concurrent.futures.TimeoutError:
            log.error(
                "Failed to shutdown server. wait_closed did not finish within timeout"
            )
        finally:
            exit(0)

    def on_draw(self):
        self._log.debug("on_draw")
        self.draw_field()

        self.draw_imgui()

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
