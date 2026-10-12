"""POSIX terminal keyboard input; polled without blocking the control loop."""

import os
import select
import sys
import termios
import tty

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.utils.hoop_input import HoopInput


@ctrl_registry.register
class TerminalKeyboardCtrl(Controller):
    def __init__(self, cfg_ctrl, env=None, device="cpu"):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)
        self.fd = sys.stdin.fileno()
        if not os.isatty(self.fd):
            raise RuntimeError("--terminal_keyboard requires an interactive TTY (SSH: use ssh -t)")
        self.original_settings = termios.tcgetattr(self.fd)
        self.reset()
        tty.setcbreak(self.fd)

    def reset(self):
        self.hoop_input = HoopInput()

    def close(self):
        if self.original_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSANOW, self.original_settings)
            self.original_settings = None

    def get_data(self):
        keys = ""
        if select.select([self.fd], [], [], 0)[0]:
            keys = os.read(self.fd, 4096).decode("ascii", errors="ignore")
        return {"keys": keys}

    def process_triggers(self, ctrl_data):
        commands = []
        for char in ctrl_data["keys"]:
            key = {
                "\r": "Key.enter",
                "\n": "Key.enter",
                "\x1b": "Key.esc",
                "\x7f": "Key.backspace",
                "\b": "Key.backspace",
                " ": "Key.space",
            }.get(char, char)
            consumed, position = self.hoop_input.feed(key)
            if position is not None:
                ctrl_data["hoop_pos_cyl"] = position
            if not consumed and key in self.triggers:
                commands.append(self.triggers[key])
        return ctrl_data, commands
