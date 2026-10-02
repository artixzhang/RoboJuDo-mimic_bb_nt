"""Keep the existing viewer's mouse controls compatible with MuJoCo >= 3.11."""

import glfw
import mujoco
import numpy as np
from mujoco_viewer import MujocoViewer as BaseMujocoViewer

MUJOCO_VERSION = tuple(map(int, mujoco.__version__.split(".")[:2]))
NEW_CAMERA_API = MUJOCO_VERSION >= (3, 11)


class MujocoViewer(BaseMujocoViewer):
    def _key_callback(self, window, key, scancode, action, mods):
        # Controller owns keyboard input. Use a callable: some glfw releases
        # wrap None as integer 0 and then try to call it on the next key event.
        pass

    def _mouse_button_callback(self, window, button, act, mods):
        self._button_left_pressed = button == glfw.MOUSE_BUTTON_LEFT and act == glfw.PRESS
        self._button_right_pressed = button == glfw.MOUSE_BUTTON_RIGHT and act == glfw.PRESS
        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x, self._last_mouse_y = int(self._scale * x), int(self._scale * y)
        now = glfw.get_time()
        self._left_double_click_pressed = False
        self._right_double_click_pressed = False
        if self._button_left_pressed:
            last = self._last_left_click_time
            self._left_double_click_pressed = last is not None and 0.01 < now - last < 0.3
            self._last_left_click_time = now
        if self._button_right_pressed:
            last = self._last_right_click_time
            self._right_double_click_pressed = last is not None and 0.01 < now - last < 0.2
            self._last_right_click_time = now

        control = bool(mods & glfw.MOD_CONTROL)
        perturb = 0
        if control and self.pert.select > 0:
            if self._button_right_pressed:
                perturb = mujoco.mjtPertBit.mjPERT_TRANSLATE
            elif self._button_left_pressed:
                perturb = mujoco.mjtPertBit.mjPERT_ROTATE
            if perturb and not self.pert.active:
                mujoco.mjv_initPerturb(self.model, self.data, self.scn, self.pert)
        self.pert.active = perturb

        if self._left_double_click_pressed or self._right_double_click_pressed:
            self._select_body(x, y, control)
            self.pert.active = 0
        if act == glfw.RELEASE:
            self.pert.active = 0

    def _select_body(self, x, y, control):
        width, height = glfw.get_window_size(self.window)
        point = np.zeros(3, dtype=np.float64)
        geom, flex, skin = (np.full(1, -1, dtype=np.int32) for _ in range(3))
        selection = [geom, flex, skin] if MUJOCO_VERSION >= (3, 0) else [geom, skin]
        body = mujoco.mjv_select(
            self.model,
            self.data,
            self.vopt,
            width / height,
            x / width,
            1 - y / height,
            self.scn,
            point,
            *selection,
        )
        if self._right_double_click_pressed:
            if body >= 0:
                self.cam.lookat[:] = point
            if control and body > 0:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self.cam.trackbodyid = body
                self.cam.fixedcamid = -1
        elif body >= 0:
            self.pert.select = body
            self.pert.skinselect = int(skin[0])
            if MUJOCO_VERSION >= (3, 0):
                self.pert.flexselect = int(flex[0])
            # Picking produces a world-space point; perturbation needs body-local coordinates.
            rotation = self.data.xmat[body].reshape(3, 3)
            self.pert.localpos[:] = rotation.T @ (point - self.data.xpos[body])
        else:
            self.pert.select = 0
            self.pert.skinselect = -1
            if MUJOCO_VERSION >= (3, 0):
                self.pert.flexselect = -1

    def _cursor_pos_callback(self, window, xpos, ypos):
        # MuJoCo 3.11 removed the scene argument from mjv_moveCamera only.
        if not NEW_CAMERA_API or self.pert.active:
            return super()._cursor_pos_callback(window, xpos, ypos)
        if not (self._button_left_pressed or self._button_right_pressed):
            return
        shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if self._button_right_pressed:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        x, y = int(self._scale * xpos), int(self._scale * ypos)
        _, height = glfw.get_framebuffer_size(window)
        with self._gui_lock:
            mujoco.mjv_moveCamera(
                self.model,
                action,
                (x - self._last_mouse_x) / height,
                (y - self._last_mouse_y) / height,
                self.cam,
            )
        self._last_mouse_x, self._last_mouse_y = x, y

    def _scroll_callback(self, window, x_offset, y_offset):
        if not NEW_CAMERA_API:
            return super()._scroll_callback(window, x_offset, y_offset)
        with self._gui_lock:
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * y_offset, self.cam)
