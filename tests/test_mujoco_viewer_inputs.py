"""Window/input regressions, enabled with ROBOJUDO_VIEWER_TESTS=1."""

import os
import unittest
from unittest.mock import Mock, patch

import glfw
import mujoco
import numpy as np

from robojudo.environment.basketball_mujoco_env import BasketballMujocoEnv
from scripts.run_basketball import make_config, parse_args


@unittest.skipUnless(os.environ.get("ROBOJUDO_VIEWER_TESTS") == "1", "Requires an explicit GUI test session")
class ViewerInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = BasketballMujocoEnv(make_config(parse_args(["--no_keyboard"])).env)
        cls.viewer = cls.env.viewer
        cls.viewer.cam.lookat[:] = cls.env.base_pos
        cls.viewer.render()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.viewer.pert.active = 0
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.viewer.cam.lookat[:] = self.env.base_pos
        self.viewer.render()

    def pick_pixel(self, robot):
        v = self.viewer
        width, height = glfw.get_window_size(v.window)
        for y in np.linspace(0.1, 0.9, 25):
            for x in np.linspace(0.1, 0.9, 25):
                point = np.zeros(3)
                indices = [np.zeros(1, dtype=np.int32) for _ in range(3)]
                body = mujoco.mjv_select(
                    v.model,
                    v.data,
                    v.vopt,
                    width / height,
                    x,
                    1 - y,
                    v.scn,
                    point,
                    *indices,
                )
                matches = body > 0 if robot else body <= 0
                if matches:
                    return x * width, y * height, body
        self.fail("No suitable picking pixel in the rendered scene")

    def double_click(self, button, x, y, mods=0):
        v = self.viewer
        v._last_left_click_time = None
        v._last_right_click_time = None
        with patch.object(glfw, "get_cursor_pos", return_value=(x, y)):
            with patch.object(glfw, "get_time", side_effect=[1.0, 1.02, 1.1]):
                v._mouse_button_callback(v.window, button, glfw.PRESS, mods)
                v._mouse_button_callback(v.window, button, glfw.RELEASE, mods)
                v._mouse_button_callback(v.window, button, glfw.PRESS, mods)
            v._mouse_button_callback(v.window, button, glfw.RELEASE, mods)
        v.render()

    def test_keyboard_events_through_glfw_poll(self):
        # Send events only to our test window, through X11 -> GLFW -> ctypes callback.
        from Xlib import XK, X, display, protocol

        v = self.viewer
        callback = glfw.set_key_callback(v.window, v._key_callback)
        self.assertTrue(callable(callback))
        received = Mock(wraps=callback)
        glfw.set_key_callback(v.window, received)
        connection = display.Display()
        try:
            window = connection.create_resource_object("window", glfw.get_x11_window(v.window))
            for name in ("a", "space", "Escape", "r", "Return", "Left", "F1"):
                keycode = connection.keysym_to_keycode(XK.string_to_keysym(name))
                for event_type, mask in (
                    (protocol.event.KeyPress, X.KeyPressMask),
                    (protocol.event.KeyRelease, X.KeyReleaseMask),
                ):
                    window.send_event(
                        event_type(
                            time=X.CurrentTime,
                            root=connection.screen().root,
                            window=window,
                            child=X.NONE,
                            root_x=0,
                            root_y=0,
                            event_x=0,
                            event_y=0,
                            state=0,
                            detail=keycode,
                            same_screen=1,
                        ),
                        event_mask=mask,
                    )
                connection.sync()
                glfw.poll_events()
            v.render()
            events = {(call.args[1], call.args[3]) for call in received.call_args_list}
            for key in (
                glfw.KEY_A,
                glfw.KEY_SPACE,
                glfw.KEY_ESCAPE,
                glfw.KEY_R,
                glfw.KEY_ENTER,
                glfw.KEY_LEFT,
                glfw.KEY_F1,
            ):
                self.assertIn((key, glfw.PRESS), events)
                self.assertIn((key, glfw.RELEASE), events)
            self.assertFalse(v._paused)
            self.assertTrue(v.is_alive)
        finally:
            glfw.set_key_callback(v.window, callback)
            connection.close()

    def test_double_click_model_background_and_tracking(self):
        v = self.viewer
        x, y, body = self.pick_pixel(robot=True)
        self.double_click(glfw.MOUSE_BUTTON_LEFT, x, y)
        self.assertEqual(v.pert.select, body)
        self.assertEqual(v.pert.skinselect, -1)
        self.assertTrue(np.isfinite(v.pert.localpos).all())
        self.double_click(glfw.MOUSE_BUTTON_RIGHT, x, y, glfw.MOD_CONTROL)
        self.assertEqual(v.cam.type, mujoco.mjtCamera.mjCAMERA_TRACKING)
        self.assertEqual(v.cam.trackbodyid, body)
        v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        v.render()
        x, y, _ = self.pick_pixel(robot=False)
        self.double_click(glfw.MOUSE_BUTTON_LEFT, x, y)
        self.assertLessEqual(v.pert.select, 0)
        self.double_click(glfw.MOUSE_BUTTON_RIGHT, x, y)
        self.assertTrue(v.is_alive)

    def test_drag_and_zoom_after_selection(self):
        v = self.viewer
        x, y, _ = self.pick_pixel(robot=True)
        self.double_click(glfw.MOUSE_BUTTON_LEFT, x, y)
        before = v.cam.azimuth
        v._button_left_pressed = True
        v._cursor_pos_callback(v.window, x + 50, y + 20)
        v._button_left_pressed = False
        self.assertNotEqual(v.cam.azimuth, before)
        distance = v.cam.distance
        v._scroll_callback(v.window, 0, 1)
        self.assertNotEqual(v.cam.distance, distance)
        v.render()


if __name__ == "__main__":
    unittest.main()
