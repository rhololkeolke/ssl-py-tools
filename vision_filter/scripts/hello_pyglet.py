#!/usr/bin/env python

import pyglet
from pyglet import gl


def main():
    window = pyglet.window.Window(width=1280, height=720, resizable=True)
    gl.glClearColor(0.133, 0.545, 0.133, 1)

    @window.event
    def on_draw():
        window.clear()

        # draw a diagonal white line
        pyglet.gl.glLineWidth(100)
        pyglet.graphics.draw(
            2, gl.GL_LINES, ("v2i", (-10000, -10000, 10000, 10000)), ("c3B", (255, 255, 255, 255, 255, 255))
        )

    pyglet.app.run()


if __name__ == "__main__":
    main()
