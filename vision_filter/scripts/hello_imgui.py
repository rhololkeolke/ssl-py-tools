#!/usr/bin/env python

import imgui
import pyglet
from imgui.integrations.pyglet import PygletRenderer
from pyglet import gl


def main():
    window = pyglet.window.Window(width=1280, height=720, resizable=True)
    gl.glClearColor(1, 1, 1, 1)
    imgui.create_context()
    impl = PygletRenderer(window)

    def update(dt, field_texture):
        imgui.new_frame()
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):

                clicked_quit, selected_quit = imgui.menu_item(
                    "Quit", "Cmd+Q", False, True
                )

                if clicked_quit:
                    exit(1)

                imgui.end_menu()
            imgui.end_main_menu_bar()

        imgui.show_test_window()

        imgui.begin("Custom window", True)
        imgui.image(field_texture.id, field_texture.width, field_texture.height, border_color=(1, 0, 0, 1))
        imgui.end()

    @window.event
    def on_draw():
        # draw a diagonal white line
        gl.glClearColor(0.133, 0.545, 0.133, 1)
        window.clear()        
        pyglet.gl.glLineWidth(100)
        pyglet.graphics.draw(
            2, gl.GL_LINES, ("v2i", (-10000, -10000, 10000, 10000)), ("c3B", (255, 255, 255, 255, 255, 255))
        )        

        # copy to texture for display inside imgui
        buffers = pyglet.image.get_buffer_manager()
        color_buffer = buffers.get_color_buffer()
        field_texture = color_buffer.get_texture()        
        
        update(1 / 60.0, field_texture)
        gl.glClearColor(1, 1, 1, 1)
        window.clear()        
        imgui.render()
        impl.render(imgui.get_draw_data())

    pyglet.app.run()
    impl.shutdown()


if __name__ == "__main__":
    main()
