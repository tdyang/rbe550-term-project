"""GUI window sizing and camera framing."""

import re
import subprocess

import pybullet as p


def detect_screen_size(default=(1920, 1080)):
    """Best-effort current-monitor resolution via xrandr, so the pybullet
    window can be opened already full-screen-sized. Needed because the mp4
    recorder locks in the window's resolution the moment it starts -- if
    the user fullscreens or resizes the window afterward, the capture
    resolution changes mid-recording and corrupts the output. Falls back
    to a common 1080p default if xrandr isn't available (e.g. non-Linux)
    or nothing is detected."""
    try:
        out = subprocess.run(["xrandr", "--current"], capture_output=True, text=True, timeout=2).stdout
        lines = [l for l in out.splitlines() if " connected " in l]
        lines.sort(key=lambda l: "primary" not in l)   # primary display first, if marked
        for l in lines:
            m = re.search(r"(\d+)x(\d+)\+\d+\+\d+", l)
            if m:
                return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    return default


def connect_gui(course_length):
    """Connects pybullet's GUI already sized to the screen (see
    detect_screen_size -- avoids a post-hoc fullscreen toggle corrupting a
    recorded video) and frames the camera to fit the whole course, looking
    down its length from one side. Must run before video logging starts so
    the very first recorded frame already has the right size and view, not
    a default close-up window that then jump-cuts or resizes mid-capture."""
    width, height = detect_screen_size()
    p.connect(p.GUI, options=f"--width={width} --height={height}")
    p.resetDebugVisualizerCamera(cameraDistance=4.5, cameraYaw=0, cameraPitch=-55,
                                  cameraTargetPosition=[course_length / 2, 0, 0])
