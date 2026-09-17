"""Why don't the viewer keys work? Run this, click the MuJoCo window, press keys.

    uv run scripts/keyprobe.py

It prints three independent facts, once a second and on every key press:

  physical   did Windows see the key at all (GetAsyncKeyState, ignores focus)
  focus      is the MuJoCo window the foreground window
  callback   did MuJoCo's key_callback fire

Press SPACE, then 1, then M, then ENTER. Ctrl-C or close the window to stop.
"""
import ctypes, ctypes.wintypes as w, os, time
import soc4180

u32 = ctypes.windll.user32
imm = ctypes.windll.imm32
pid = os.getpid()
hits = []

def on_key(kc):
    hits.append(kc)
    print(f"  CALLBACK  keycode {kc}", flush=True)

def our_window():
    out = []
    P = ctypes.WINFUNCTYPE(ctypes.c_bool, w.HWND, w.LPARAM)
    def cb(h, l):
        p = w.DWORD(); u32.GetWindowThreadProcessId(h, ctypes.byref(p))
        if p.value == pid and u32.IsWindowVisible(h): out.append(h)
        return True
    u32.EnumWindows(P(cb), 0)
    return out[0] if out else None

def title(h):
    n = u32.GetWindowTextLengthW(h)
    b = ctypes.create_unicode_buffer(n + 1)
    u32.GetWindowTextW(h, b, n + 1)
    return b.value

class GUITHREADINFO(ctypes.Structure):
    _fields_ = [("cbSize", w.DWORD), ("flags", w.DWORD), ("hwndActive", w.HWND),
                ("hwndFocus", w.HWND), ("hwndCapture", w.HWND),
                ("hwndMenuOwner", w.HWND), ("hwndMoveSize", w.HWND),
                ("hwndCaret", w.HWND), ("rcCaret", w.RECT)]


def focus_window():
    """The window that actually receives WM_KEYDOWN, which is NOT the foreground
    window: a window can be foreground while keyboard focus sits elsewhere."""
    g = GUITHREADINFO()
    g.cbSize = ctypes.sizeof(g)
    if u32.GetGUIThreadInfo(0, ctypes.byref(g)):
        return g.hwndFocus
    return None


WATCH = {0x20: "SPACE", 0x0D: "ENTER", 0x25: "LEFT", 0x27: "RIGHT",
         **{0x30 + d: str(d) for d in range(10)},
         0x4D: "M", 0x52: "R", 0x41: "A"}

model = soc4180.load_g1()
data = soc4180.keyframe_data(model, "stand")
print(__doc__)

with soc4180.launch_viewer(model, data, passive=True, key_callback=on_key) as v:
    hwnd = None
    for _ in range(40):
        hwnd = our_window()
        if hwnd:
            break
        time.sleep(0.25)
    print(f"MuJoCo window: {hwnd}  {title(hwnd) if hwnd else ''}\n", flush=True)

    last_beat, down = 0.0, set()
    while v.is_running():
        for vk, name in WATCH.items():
            pressed = bool(u32.GetAsyncKeyState(vk) & 0x8000)
            if pressed and vk not in down:
                down.add(vk)
                fg = u32.GetForegroundWindow()
                n_before = len(hits)
                time.sleep(0.15)                       # let the event land
                fired = len(hits) > n_before
                fw = focus_window()
                print(f"  PHYSICAL  {name:6s} fg={'MuJoCo' if fg == hwnd else repr(title(fg))}"
                      f"  keyfocus={'MuJoCo' if fw == hwnd else ('none' if not fw else repr(title(fw)))}"
                      f"  callback={'YES' if fired else 'NO  <-- swallowed'}", flush=True)
            elif not pressed:
                down.discard(vk)

        if time.time() - last_beat > 2.0:
            last_beat = time.time()
            fg = u32.GetForegroundWindow()
            tid = u32.GetWindowThreadProcessId(fg, None)
            layout = u32.GetKeyboardLayout(tid)
            himc = imm.ImmGetContext(fg)
            ime_open, conv = "n/a", "n/a"
            if himc:
                ime_open = bool(imm.ImmGetOpenStatus(himc))
                c, s = w.DWORD(), w.DWORD()
                if imm.ImmGetConversionStatus(himc, ctypes.byref(c), ctypes.byref(s)):
                    conv = hex(c.value)      # 0x0 = alphanumeric, 0x1 = Hangul
                imm.ImmReleaseContext(fg, himc)
            fw = focus_window()
            print(f"  [state] foreground={'MuJoCo' if fg == hwnd else repr(title(fg))}"
                  f"  keyfocus={'MuJoCo' if fw == hwnd else ('none' if not fw else repr(title(fw)))}"
                  f"  layout={hex(layout & 0xFFFF)}  IME_open={ime_open}  conversion={conv}"
                  f"  keys_seen={len(hits)}", flush=True)
        time.sleep(0.02)
        v.sync()
print("\nkeycodes the callback received:", hits)
