"""Controller events and TTY input tested offline, without a display or robot."""

import importlib
import os
import pty
import select
import sys
import termios
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

from robojudo.controller.joystick_ctrl import JoystickCtrl
from robojudo.controller.terminal_keyboard_ctrl import TerminalKeyboardCtrl
from robojudo.controller.utils.hoop_input import HoopInput
from scripts.run_basketball import gamepad_triggers, make_config, parse_args


class InputTests(unittest.TestCase):
    def test_hoop_editor_consumes_digits_and_supports_backspace(self):
        editor = HoopInput()
        for key in ("h", "3", "Key.space", "-", "0", ".", "2", "Key.space", "2", "Key.backspace", "1", ".", "8"):
            consumed, position = editor.feed(key)
            self.assertTrue(consumed)
            self.assertIsNone(position)
        self.assertEqual(editor.feed("Key.enter"), (True, [3, -0.2, 1.8]))
        self.assertEqual(editor.feed("r"), (False, None))
        for text in ("nan 0 1", "-1 0 1", "1 2", "1 2 3 4"):
            for char in "h" + text:
                editor.feed(char)
            self.assertEqual(editor.feed("Key.enter"), (True, None))
        editor.feed("h")
        self.assertEqual(editor.feed("Key.esc"), (False, None))
        self.assertIsNone(editor.text)

    def test_gamepad_combos_do_not_trigger_reset_or_stop(self):
        ctrl = object.__new__(JoystickCtrl)
        ctrl.triggers = gamepad_triggers("Select")
        ctrl.combination_init_buttons = ["R2"]
        ctrl.onhold_buttons = set()

        def send(name, pressed=True):
            return ctrl.process_triggers({"button_event": [{"type": "button", "name": name, "pressed": pressed}]})[1]

        self.assertEqual(send("R2"), [])
        for index, button in enumerate(("A", "B", "X", "Y", "Up", "Down", "Left", "Right")):
            self.assertEqual(send(button), [f"[SKILL_{index}]"])
            send(button, False)
        self.assertEqual(send("Select"), ["[SHUTDOWN]"])
        send("R2", False)
        self.assertEqual(send("Y"), ["[MOTION_RESET]"])
        self.assertEqual(send("Start"), ["[POLICY_START]"])
        self.assertEqual(send("A"), [])

    def test_keyboard_press_release_and_text_are_isolated(self):
        with patch.dict(sys.modules, {"pynput": SimpleNamespace(keyboard=object())}):
            module = importlib.import_module("robojudo.controller.keyboard_ctrl")
        ctrl = object.__new__(module.KeyboardCtrl)
        ctrl.cfg_ctrl = make_config(parse_args(["--real"])).ctrl[0]
        ctrl.triggers = ctrl.cfg_ctrl.triggers
        ctrl.event_queue = Queue()
        ctrl.reset()

        def send(keys):
            events = [{"type": "keyboard", "name": name, "pressed": pressed} for name, pressed in keys]
            return ctrl.process_triggers({"keyboard_event": events})

        self.assertEqual(send([("2", True), ("2", True)])[1], ["[SKILL_1]"])
        send([("2", False)])
        events = []
        for key in ("h", "3", "Key.space", "0", "Key.space", "2", "Key.enter"):
            events.extend([(key, True), (key, False)])
        data, commands = send(events)
        self.assertEqual(commands, [])
        self.assertEqual(data["hoop_pos_cyl"], [3, 0, 2])
        self.assertEqual(send([("r", True)])[1], ["[MOTION_RESET]"])
        self.assertEqual(send([("h", True), ("Key.esc", True)])[1], ["[SHUTDOWN]"])

    def test_terminal_is_nonblocking_and_restores_tty(self):
        master, slave = pty.openpty()
        self.addCleanup(os.close, master)
        self.addCleanup(os.close, slave)
        settings = termios.tcgetattr(slave)
        cfg = make_config(parse_args(["--real", "--terminal_keyboard"])).ctrl[0]
        with os.fdopen(os.dup(slave)) as stream, patch("sys.stdin", stream):
            ctrl = TerminalKeyboardCtrl(cfg)
            try:
                self.assertEqual(ctrl.get_data()["keys"], "")
                os.write(master, b"h4 0.2 2\r1r")
                # select waits only in the test, never inside the control loop.
                self.assertTrue(select.select([slave], [], [], 1)[0])
                data, commands = ctrl.process_triggers(ctrl.get_data())
                self.assertEqual(data["hoop_pos_cyl"], [4, 0.2, 2])
                self.assertEqual(commands, ["[SKILL_0]", "[MOTION_RESET]"])
            finally:
                ctrl.close()
            self.assertEqual(termios.tcgetattr(slave), settings)

    def test_controller_initialization_failure_restores_previous_controller(self):
        from robojudo.controller.ctrl_manager import CtrlManager
        cfg = make_config(parse_args(["--real", "--terminal_keyboard"]))
        terminal = Mock()
        with patch("robojudo.controller.ctrl_manager.getattr", create=True) as lookup:
            lookup.side_effect = [Mock(return_value=terminal), Mock(side_effect=RuntimeError("failure")), terminal.close]
            with self.assertRaisesRegex(RuntimeError, "failure"):
                CtrlManager(cfg.ctrl)
        terminal.close.assert_called_once()

    def test_keyboard_and_gamepad_configs_coexist(self):
        cfg = make_config(parse_args(["--real", "--terminal_keyboard"]))
        self.assertEqual([c.ctrl_type for c in cfg.ctrl], ["TerminalKeyboardCtrl", "UnitreeCtrl"])
        cfg = make_config(parse_args(["--joystick"]))
        self.assertEqual([c.ctrl_type for c in cfg.ctrl], ["KeyboardCtrl", "JoystickCtrl"])


if __name__ == "__main__":
    unittest.main()
