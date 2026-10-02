"""Renders PNG snapshots of the robot, ball, and obstacle course. Uses
PyBullet's offscreen TinyRenderer (p.DIRECT, no GUI/display needed) and the
real course generator/robot/physics modules -- not a hand-drawn schematic --
so the images match exactly what the live simulation looks like.

Three views:
  robot    Just the chassis/arm/paddle/ball assembly, no obstacles or
           course, at a neutral pose.
  hero     A close-up of the robot/arm/ball among nearby obstacles at some
           point along a solved course, camera auto-aimed perpendicular to
           the arm so chassis, arm, paddle, and ball don't line up back-to-
           front behind each other.
  overhead A straight-down view of the whole course: every obstacle, the
           course boundary, the planned robot-body path, and the robot/ball
           at the same point the hero shot uses.

Debug lines (the arm, the path, the boundary, all drawn via
addUserDebugLine elsewhere in this project) are a GUI-only overlay that
getCameraImage never captures, so this script stands in real thin-box
geometry for all three wherever a snapshot needs them.

Saving goes through a .ppm intermediate (pybullet's raw pixel buffer, no
extra Python image library needed) converted to .png with ImageMagick's
`convert`, which must be on PATH.

Run:
    python render_scene.py --view robot [--out figures/robot.png] [--phi DEG]
    python render_scene.py --view hero [--seed N] [--out figures/hero.png] [--progress F]
    python render_scene.py --view overhead [--seed N] [--out figures/overhead.png]
"""

import argparse
import math
import os
import random
import subprocess

import numpy as np
import pybullet as p
import pybullet_data

from config import ARM_LENGTH, BASE_XY_HALF, CONTACT_Z, COURSE_HALF_WIDTH, COURSE_LENGTH, PADDLE_CENTER_Z
from obstacles import spawn_obstacles
from physics import spawn_ball
from planning import generate_solvable_course
from robot import Robot


def spawn_thin_box(p0, p1, z, half_thickness, rgba):
    """A static thin box from p0 to p1 at height z -- used to stand in for
    whatever this project normally draws as a GUI-only debug line, so it
    actually shows up in an offscreen render."""
    p0, p1 = np.array(p0), np.array(p1)
    length = float(np.linalg.norm(p1 - p0))
    if length < 1e-9:
        return
    mid = (p0 + p1) / 2
    yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    half = [length / 2, half_thickness, half_thickness]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba)
    quat = p.getQuaternionFromEuler([0, 0, yaw])
    p.createMultiBody(0, col, vis, [mid[0], mid[1], z], quat)


def spawn_visible_arm(base_xy, paddle_xy, phi):
    unit = np.array([math.cos(phi), math.sin(phi)])
    base_face = np.array(base_xy) - BASE_XY_HALF * unit
    spawn_thin_box(base_face, paddle_xy, PADDLE_CENTER_Z, 0.02, [0.2, 0.7, 0.3, 1])


def spawn_visible_path(path_xs, path_ys):
    for i in range(len(path_xs) - 1):
        spawn_thin_box((path_xs[i], path_ys[i]), (path_xs[i + 1], path_ys[i + 1]), 0.01, 0.015, [1, 1, 0, 1])


def spawn_boundary():
    corners = [(0, -COURSE_HALF_WIDTH), (COURSE_LENGTH, -COURSE_HALF_WIDTH),
               (COURSE_LENGTH, COURSE_HALF_WIDTH), (0, COURSE_HALF_WIDTH)]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        spawn_thin_box(a, b, 0.01, 0.01, [0.1, 0.1, 0.1, 1])


def build_robot_only(phi):
    """Just the chassis/arm/paddle/ball, no obstacles or course -- ball at
    the world origin, chassis offset from it by the arm at angle phi."""
    ball_xy = (0.0, 0.0)
    chassis_xy = (ARM_LENGTH * math.cos(phi), ARM_LENGTH * math.sin(phi))

    ball = spawn_ball()
    p.resetBasePositionAndOrientation(ball, [ball_xy[0], ball_xy[1], CONTACT_Z], [0, 0, 0, 1])
    robot = Robot()
    robot.set_pose(ball_xy, phi)
    spawn_visible_arm(robot.base_xy, ball_xy, phi)

    return chassis_xy, ball_xy


def build_scene(seed, progress, draw_path=False, draw_boundary=False):
    rng = random.Random(seed)
    obstacles, path, phis_list, attempts = generate_solvable_course(rng)
    path_xs = np.array([w[0] for w in path])
    path_ys = np.array([w[1] for w in path])
    path_phis = np.array(phis_list)

    idx = int(np.clip(progress, 0.0, 1.0) * (len(path) - 1))
    chassis_xy = (float(path_xs[idx]), float(path_ys[idx]))
    phi = float(path_phis[idx])
    ball_xy = (chassis_xy[0] - ARM_LENGTH * math.cos(phi), chassis_xy[1] - ARM_LENGTH * math.sin(phi))

    spawn_obstacles(obstacles)
    if draw_path:
        spawn_visible_path(path_xs, path_ys)
    if draw_boundary:
        spawn_boundary()

    ball = spawn_ball()
    p.resetBasePositionAndOrientation(ball, [ball_xy[0], ball_xy[1], CONTACT_Z], [0, 0, 0, 1])
    robot = Robot()
    robot.set_pose(ball_xy, phi)
    spawn_visible_arm(robot.base_xy, ball_xy, phi)

    return len(obstacles), attempts, chassis_xy, ball_xy, phi


def save_png(width, height, rgb_pixels, out_path):
    frame = np.reshape(rgb_pixels, (height, width, 4))[:, :, :3].astype(np.uint8)
    ppm_path = out_path.rsplit(".", 1)[0] + ".ppm"
    with open(ppm_path, "wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        f.write(frame.tobytes())
    subprocess.run(["convert", ppm_path, out_path], check=True)
    os.remove(ppm_path)


def side_camera_eye(mid, phi, side, standoff, height_cam):
    """Stand the camera off to the side of the arm, not along it: along the
    arm, the chassis/arm/paddle/ball all line up back-to-front and hide
    behind each other. The arm direction is unit(phi) (chassis - ball);
    rotating it 90 degrees in-plane gives a direction straight out the
    robot's side, which is what we actually want to stand on."""
    d = np.array([math.cos(phi), math.sin(phi)])
    perp = np.array([-d[1], d[0]]) if side == "left" else np.array([d[1], -d[0]])
    return [mid[0] + perp[0] * standoff, mid[1] + perp[1] * standoff, height_cam]


def render_from(width, height, eye, target):
    view_mat = p.computeViewMatrix(eye, target, [0, 0, 1])
    proj_mat = p.computeProjectionMatrixFOV(fov=60, aspect=width / height, nearVal=0.05, farVal=25)
    width, height, rgb, _depth, _seg = p.getCameraImage(width, height, view_mat, proj_mat, renderer=p.ER_TINY_RENDERER)
    return width, height, rgb


def render(width, height, target, distance, yaw, pitch):
    view_mat = p.computeViewMatrixFromYawPitchRoll(target, distance, yaw, pitch, 0, upAxisIndex=2)
    proj_mat = p.computeProjectionMatrixFOV(fov=60, aspect=width / height, nearVal=0.05, farVal=25)
    width, height, rgb, _depth, _seg = p.getCameraImage(width, height, view_mat, proj_mat, renderer=p.ER_TINY_RENDERER)
    return width, height, rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--view", choices=["robot", "hero", "overhead"], default="robot")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--progress", type=float, default=0.45,
                         help="0-1 fraction along the planned path to pose the robot/ball at")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--distance", type=float, default=None, help="Camera distance, meters (view-specific default)")
    parser.add_argument("--pitch", type=float, default=None, help="Camera pitch, degrees (overhead only)")
    parser.add_argument("--yaw", type=float, default=None, help="Camera yaw, degrees (overhead only)")
    parser.add_argument("--height-cam", type=float, default=1.3, help="Camera height off the ground (robot/hero)")
    parser.add_argument("--side", choices=["left", "right"], default="left",
                         help="Which side of the arm the camera stands on (robot/hero)")
    parser.add_argument("--phi", type=float, default=90,
                         help="Arm angle in degrees, chassis relative to ball (robot view only)")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randrange(2 ** 31)
    print(f"seed: {seed}")

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    if args.view == "robot":
        phi = math.radians(args.phi)
        chassis_xy, ball_xy = build_robot_only(phi)
        mid = np.array([(chassis_xy[0] + ball_xy[0]) / 2, (chassis_xy[1] + ball_xy[1]) / 2])
        target = [mid[0], mid[1], 0.35]
        standoff = args.distance if args.distance is not None else 1.1
        eye = side_camera_eye(mid, phi, args.side, standoff, args.height_cam)
        out_path = args.out or "figures/robot.png"
        width, height, rgb = render_from(args.width, args.height, eye, target)
    elif args.view == "hero":
        n_obs, attempts, chassis_xy, ball_xy, phi = build_scene(seed, args.progress)
        print(f"{n_obs} obstacles, solved after {attempts} attempt(s)")
        mid = np.array([(chassis_xy[0] + ball_xy[0]) / 2, (chassis_xy[1] + ball_xy[1]) / 2])
        target = [mid[0], mid[1], 0.35]
        standoff = args.distance if args.distance is not None else 1.8
        eye = side_camera_eye(mid, phi, args.side, standoff, args.height_cam)
        out_path = args.out or "figures/hero.png"
        width, height, rgb = render_from(args.width, args.height, eye, target)
    else:
        n_obs, attempts, chassis_xy, ball_xy, phi = build_scene(
            seed, args.progress, draw_path=True, draw_boundary=True)
        print(f"{n_obs} obstacles, solved after {attempts} attempt(s)")
        target = [COURSE_LENGTH / 2, 0, 0]
        distance = args.distance if args.distance is not None else 5.3
        pitch = args.pitch if args.pitch is not None else -89.9
        yaw = args.yaw if args.yaw is not None else 0
        out_path = args.out or "figures/overhead.png"
        width, height, rgb = render(args.width, args.height, target, distance, yaw, pitch)
    p.disconnect()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_png(width, height, rgb, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
