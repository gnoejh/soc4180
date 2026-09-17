"""Type viewer commands into the terminal when the window will not take keys.

MuJoCo's viewer receives key presses through GLFW, and on Windows that path can
be broken by things outside our control: an IME that swallows letters while it
is composing, a keyboard hook, or keyboard focus sitting on a window other than
the one in front. A lab class on thirty unknown laptops cannot depend on it.

``terminal_keys`` reads the console the script was launched from and feeds the
same ``key_callback`` the viewer would have called, so every command has a
second route that needs no window focus at all::

    on_key = ...                       # the callback you already wrote
    soc4180.terminal_keys(on_key)      # start the reader, then open the viewer

Pressing ``1`` in the terminal is exactly equivalent to pressing ``1`` in the
window: single keys, no enter needed, arrows included. ``q`` or ctrl-C stops.

When stdin is not a console (piped input, a test) it falls back to reading whole
lines, so ``1 m`` on one line still works.
"""

from __future__ import annotations

import sys
import threading
from _thread import interrupt_main

# GLFW keycodes for the non-printable keys the lab scripts bind. Printable keys
# need no table: GLFW numbers them by their *uppercase* ASCII codepoint.
NAMED = {
    "space": 32,
    "enter": 257,
    "return": 257,
    "left": 263,
    "right": 262,
    "up": 265,
    "down": 264,
    "tab": 258,
    "esc": 256,
}


def keycode(token: str) -> int | None:
    """The GLFW keycode a typed word stands for, or None if it is not a key."""
    token = token.strip()
    if not token:
        return NAMED["enter"]
    if token.lower() in NAMED:
        return NAMED[token.lower()]
    if len(token) == 1 and token.isprintable():
        return ord(token.upper())
    return None


def _emit(key_callback, code) -> None:
    if code is not None:
        key_callback(code)


def _read_lines(key_callback, quit_tokens) -> None:
    """Whole-line fallback: used when stdin is a pipe rather than a console."""
    while True:
        try:
            line = sys.stdin.readline()
        except (ValueError, OSError):            # stdin closed under us
            return
        if not line:                             # EOF
            return
        if line.strip().lower() in quit_tokens:
            print("  [terminal] stopped reading keys", flush=True)
            return
        for token in (line.split() or [""]):
            code = keycode(token)
            if code is None:
                print(f"  [terminal] {token!r} is not a key", flush=True)
            else:
                key_callback(code)


# Arrow keys arrive as an escape sequence rather than a character, and the
# prefix differs per platform: Windows sends 0x00/0xe0 then a letter, POSIX
# terminals send ESC [ then a letter.
NEWLINES = (chr(13), chr(10))
_WINDOWS_ARROWS = {"H": 265, "P": 264, "K": 263, "M": 262}
_POSIX_ARROWS = {"A": 265, "B": 264, "D": 263, "C": 262}


def _char_code(ch: str) -> int | None:
    """One typed character as a GLFW keycode. Space and enter are characters
    here, not words, so they cannot go through keycode()'s token parsing."""
    if ch in NEWLINES:
        return NAMED["enter"]
    if ch == " ":
        return NAMED["space"]
    return keycode(ch)


def _read_keys_windows(key_callback, quit_tokens) -> None:
    import msvcrt

    while True:
        ch = msvcrt.getwch()
        if ch in (chr(0), chr(0xE0)):                # arrow or function key
            _emit(key_callback, _WINDOWS_ARROWS.get(msvcrt.getwch()))
            continue
        if ch == chr(3):                          # ctrl-C must still stop the lab
            interrupt_main()
            return
        if ch.lower() in quit_tokens:
            print("  [terminal] stopped reading keys", flush=True)
            return
        _emit(key_callback, _char_code(ch))


def _read_keys_posix(key_callback, quit_tokens) -> None:
    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1)
            if not ch:
                return
            if ch == chr(27):                      # ESC [ A .. D
                if sys.stdin.read(1) == "[":
                    _emit(key_callback, _POSIX_ARROWS.get(sys.stdin.read(1)))
                continue
            if ch.lower() in quit_tokens:
                print("  [terminal] stopped reading keys", flush=True)
                return
            _emit(key_callback, _char_code(ch))
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def terminal_keys(key_callback, *, quit_tokens=("q", "quit", "exit")) -> threading.Thread:
    """Feed terminal key presses to ``key_callback`` as if they were viewer keys.

    Starts a daemon thread and returns it. On a console each key acts as soon as
    it is pressed -- no enter -- including the arrow keys; ``q`` or ctrl-C stops
    the reader. With piped input it reads lines instead, so a line like ``1 m``
    sends both keys. The thread is a daemon, so it never keeps the process alive
    after the viewer window closes.
    """

    try:
        interactive = sys.stdin is not None and sys.stdin.isatty()
    except (ValueError, AttributeError):
        interactive = False

    if not interactive:
        target = _read_lines
    elif sys.platform == "win32":
        target = _read_keys_windows
    else:
        target = _read_keys_posix

    def read() -> None:
        try:
            target(key_callback, {t.lower() for t in quit_tokens})
        except (ValueError, OSError):              # stdin closed under us
            return

    thread = threading.Thread(target=read, daemon=True, name="terminal-keys")
    thread.start()
    return thread
