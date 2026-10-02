"""Renders a reference image of the three obstacle kinds (crate, pillar,
wall) side by side, each drawn at the midpoint of its random size range
from obstacles.OBSTACLE_SIZE_RANGES -- the single source of truth
generate_course() itself draws from, so this can't silently drift out of
sync with what actually gets spawned in a course. Also prints the exact
min-max sizing table (and, with --latex, ready-to-paste LaTeX rows).

Uses PyBullet's offscreen TinyRenderer (p.DIRECT, no GUI/display needed).
Saving goes through a .ppm intermediate converted to .png with
ImageMagick's `convert` (also used to stamp the per-object caption), which
must be on PATH.

Run:
    python render_obstacle_types.py [--out figures/obstacle_types.png] [--latex]
"""

import argparse
import os
import subprocess

import numpy as np
import pybullet as p
import pybullet_data

from obstacles import Obstacle, OBSTACLE_SIZE_RANGES, spawn_obstacles

SPACING = 1.4   # meters between each example obstacle's center, placed along x


def midpoint(lo_hi):
    lo, hi = lo_hi
    return (lo + hi) / 2


def representative_obstacles():
    """One Obstacle per kind, at the midpoint size of its range, yaw=0 (so
    hx/hy align with world x/y and read naturally left-to-right), spaced
    out along x in OBSTACLE_SIZE_RANGES's own order."""
    obstacles = []
    for i, (kind, ranges) in enumerate(OBSTACLE_SIZE_RANGES.items()):
        hx = midpoint(ranges["hx"])
        hy = hx if kind == "pillar" else midpoint(ranges["hy"])
        height = midpoint(ranges["height"])
        obstacles.append(Obstacle(kind, i * SPACING, 0.0, 0.0, hx, hy, height))
    return obstacles


def sizing_rows():
    rows = []
    for kind, r in OBSTACLE_SIZE_RANGES.items():
        if kind == "pillar":
            dims = f"radius {r['hx'][0]:.2f}-{r['hx'][1]:.2f} m"
        else:
            dims = f"{2*r['hx'][0]:.2f}-{2*r['hx'][1]:.2f} m (x) x {2*r['hy'][0]:.2f}-{2*r['hy'][1]:.2f} m (y)"
        rows.append((kind, dims, f"{r['height'][0]:.2f}-{r['height'][1]:.2f} m"))
    return rows


def print_table():
    print(f"{'kind':<8} {'footprint (full, not half-extent)':<42} {'height':<12}")
    for kind, dims, height in sizing_rows():
        print(f"{kind:<8} {dims:<42} {height:<12}")


def print_latex_rows():
    print("\n% LaTeX tabular rows (kind & footprint & height):")
    for kind, dims, height in sizing_rows():
        print(f"{kind.capitalize():<8} & {dims} & {height} \\\\")


def save_png(width, height, rgb_pixels, out_path):
    frame = np.reshape(rgb_pixels, (height, width, 4))[:, :, :3].astype(np.uint8)
    ppm_path = out_path.rsplit(".", 1)[0] + ".ppm"
    with open(ppm_path, "wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        f.write(frame.tobytes())
    subprocess.run(["convert", ppm_path, out_path], check=True)
    os.remove(ppm_path)


SHAPE_WORD = {"crate": "box", "pillar": "cylinder", "wall": "box"}
COLUMN_X = {"crate": 160, "pillar": 800, "wall": 1320}   # tuned for main()'s fixed camera framing


def add_captions(out_path):
    """Appends a white strip under the render and stamps one caption per
    obstacle kind, left-to-right in the same order they're laid out in the
    scene, built from the same sizing_rows() the printed table uses so the
    image can't say something different from the numbers next to it.
    Column x-positions are tuned for the fixed camera framing main() uses,
    not computed from the camera matrices, so if that framing changes
    these will need re-tuning too."""
    cmd = ["convert", out_path, "-gravity", "South", "-background", "white", "-splice", "0x210"]
    for kind, footprint, height in sizing_rows():
        x = COLUMN_X[kind]
        desc = f"{SHAPE_WORD[kind]}, {footprint},\nheight {height}"
        cmd += ["-gravity", "NorthWest", "-pointsize", "26", "-font", "Helvetica-Bold",
                "-annotate", f"+{x}+915", kind.capitalize()]
        cmd += ["-gravity", "NorthWest", "-pointsize", "18", "-font", "Helvetica",
                "-annotate", f"+{x}+955", desc]
    cmd.append(out_path)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=str, default="figures/obstacle_types.png")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--no-caption", action="store_true", help="Skip the stamped caption strip")
    args = parser.parse_args()

    print_table()
    if args.latex:
        print_latex_rows()

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    spawn_obstacles(representative_obstacles())

    target = [SPACING, 0, 0.2]
    view_mat = p.computeViewMatrixFromYawPitchRoll(target, 3.6, 0, -18, 0, upAxisIndex=2)
    proj_mat = p.computeProjectionMatrixFOV(fov=55, aspect=args.width / args.height, nearVal=0.05, farVal=25)
    width, height, rgb, _depth, _seg = p.getCameraImage(
        args.width, args.height, view_mat, proj_mat, renderer=p.ER_TINY_RENDERER)
    p.disconnect()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_png(width, height, rgb, out_path)
    if not args.no_caption:
        add_captions(out_path)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
