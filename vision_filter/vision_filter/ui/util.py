import structlog
from pyglet import gl

RAD2DEG = 57.29577951308232


class Transform:
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1, 1)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)

        self._is_enabled = False

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disable()

    def enable(self):
        if not self._is_enabled:
            self._is_enabled = True
            gl.glPushMatrix()
            gl.glTranslatef(
                self.translation[0], self.translation[1], 0
            )  # translate to GL loc ppint
            gl.glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
            gl.glScalef(self.scale[0], self.scale[1], 1)
        else:
            structlog.get_logger().warning("enable called on already enabled transform")

    def disable(self):
        if self._is_enabled:
            self._is_enabled = False
            gl.glPopMatrix()
        else:
            structlog.get_logger().warning(
                "disable called on already disabled transform"
            )

    @property
    def is_enabled(self):
        return self._is_enabled

    def set_translation(self, newx, newy):
        self.translation = (float(newx), float(newy))

    def set_rotation(self, new):
        self.rotation = float(new)

    def set_scale(self, newx, newy):
        self.scale = (float(newx), float(newy))
