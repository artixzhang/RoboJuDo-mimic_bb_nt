"""Nonblocking keyboard text editor for a cylindrical hoop position."""

import logging
import math

logger = logging.getLogger(__name__)


class HoopInput:
    def __init__(self):
        self.text = None

    def feed(self, key):
        """Return (consumed, completed_position); Esc remains an emergency stop."""
        if key == "Key.esc":
            self.text = None
            return False, None
        if self.text is None:
            if key.lower() == "h":
                self.text = ""
                logger.info("Hoop input: type r theta z (m, rad, m), then Enter; Backspace edits")
                return True, None
            return False, None
        if key == "Key.enter":
            text, self.text = self.text, None
            try:
                values = [float(v) for v in text.replace(",", " ").split()]
                if len(values) != 3 or not all(math.isfinite(v) for v in values) or values[0] < 0:
                    raise ValueError
            except ValueError:
                logger.warning("Invalid hoop input %r: expected finite r theta z, r >= 0", text)
                return True, None
            return True, values
        if key == "Key.backspace":
            self.text = self.text[:-1]
        elif key == "Key.space":
            self.text += " "
        elif len(key) == 1 and len(self.text) < 128:
            self.text += key
        return True, None
