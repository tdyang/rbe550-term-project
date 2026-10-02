"""
Court Vision — Objective 3 scaffold, v3
Static obstacle course: a randomly generated course of 6-9 common obstacles
(crates, pillars, walls), each randomly placed and oriented. The ball and
robot move together like a player dribbling through a crowd: the planned
path follows the robot body's own center, curving to slip through gaps,
with the ball/paddle trailing at a fixed arm's length -- defaulting to
beside the robot but falling in directly behind it wherever the side is
too tight -- not locked to a rigid heading+90-degree offset. Heading only
changes where a straight line wouldn't fit.

This replaces an earlier design where the ball's own point was the planned
path and the chassis was the thing offset from it; planning around the
robot body directly means the chassis -- the part that actually has to
clear every obstacle by its full footprint -- is what the search optimizes
for, with the ball's own target path derived from it afterward.

Pipeline:
  1. obstacles.generate_course(): random obstacles (kind, position, size,
     yaw), scattered on both sides of the straight-ahead line, spaced far
     enough apart to leave room for the robot body *and* the ball/arm's
     swing through any gap between them, not just the chassis alone.
  2. planning.hybrid_astar(): a state-lattice search over (x, y, heading)
     -- not just (x, y) -- since the robot body needs both position *and*
     orientation to route through a gap only wide enough at an angle. Each
     step is a short forward arc at one of a few discrete turning rates.
     Both the robot body's own point and the existence of *some*
     wide-enough arm angle (clearing the ball/paddle end) are checked for
     clearance at every step.
  3. planning.plan_phi_sequence(): a second search, over the already-found
     path, that picks a specific arm angle at every point and verifies
     every transition between consecutive points is fully collision-free
     -- not just that some angle works at each point in isolation, which
     doesn't guarantee the ball/paddle can actually get from one point's
     safe angle to the next's. This is what makes the run free of live
     emergency corrections: the whole arm trajectory is proven traversable
     before the simulation ever starts, not discovered (and occasionally
     failed to find) reactively frame by frame.
  If a random course turns out unsolvable at either stage, generation
  retries with a fresh course rather than failing.

  main() below then ties it together: spawns the pybullet world
  (obstacles, planned-path debug line, ball, robot), and runs the
  dribble-and-push control loop (physics.py, robot.py) frame by frame.

Run:
    pip install numpy pybullet
    python simulation.py [--seed N] [--video out.mp4]

    --video saves under term_project/video_runs/ (created if missing) --
    pass an absolute path instead to save elsewhere. Requires ffmpeg on
    PATH (pybullet shells out to it to encode the MP4) and records the
    whole GUI window from course spawn to finish.
"""

import argparse
import math
import os
import random
import time

import numpy as np
import pybullet as p
import pybullet_data

from config import ARM_LENGTH, COURSE_HALF_WIDTH, COURSE_LENGTH, DT, FLOOR_E, LANE_VX, PADDLE_CENTER_Z
from display import connect_gui
from obstacles import spawn_obstacles
from physics import BallisticModel, ContactEvent, ContactTarget, mirror_law_push, spawn_ball
from planning import angle_diff_signed, generate_solvable_course
from robot import Robot

VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_runs")

PUSH_DEPTH, PUSH_STEPS = 0.05, 12


def plan_next_contact(paddle_xs, paddle_ys, ball_xy, wp_idx, t_now, bounce: BallisticModel, lane_speed=LANE_VX):
    """Waypoint-follower, not a nearest-point search: aims at the ball's own
    target point paddle_xs/ys[wp_idx] -- derived from the planned robot-body
    path offset by the arm -- advancing wp_idx (monotonically -- it never
    goes back) once the ball's actual position is within one dribble
    cycle's travel of it. Two earlier designs failed here: an accumulated
    distance-traveled counter overstates true progress whenever the ball's
    straight chords cut a corner, a self-reinforcing drift; and a nearest-
    point *search* can snap backward to an earlier part of the path that a
    sharp turn brings spatially close to, oscillating in place forever.
    Aiming at one specific, monotonically-advancing lattice point sidesteps
    both -- there's nothing to search or accumulate, just "close enough,
    move on" -- while still re-deriving the aim direction from the ball's
    real position every call, so physics-level drift never compounds."""
    step_dist = lane_speed * bounce.flight_time
    while wp_idx < len(paddle_xs) - 1 and math.hypot(paddle_xs[wp_idx] - ball_xy[0], paddle_ys[wp_idx] - ball_xy[1]) < step_dist:
        wp_idx += 1
    target_xy = np.array([paddle_xs[wp_idx], paddle_ys[wp_idx]])
    delta = target_xy - np.array(ball_xy)
    dist = np.linalg.norm(delta)
    push_dir = delta / dist if dist > 1e-9 else np.array([1.0, 0.0])
    return ContactTarget(xy=target_xy, t=t_now + bounce.flight_time, push_dir=push_dir), wp_idx


def phi_at_progress(path_xs, path_ys, path_phis, ball_xy, wp_idx):
    """Arm angle for the ball's current position along the (wp_idx-1 ->
    wp_idx) segment of the precomputed, verified phi sequence -- position
    and arm angle interpolated together, exactly matching how
    plan_phi_sequence validated that transition. Called with the ball's own
    target path (paddle_xs/ys), not the robot-body path, since that's the
    array the ball's real position actually tracks. No live safety check
    needed: the whole line was proven collision-free in advance."""
    i0, i1 = max(wp_idx - 1, 0), wp_idx
    x0, y0 = path_xs[i0], path_ys[i0]
    x1, y1 = path_xs[i1], path_ys[i1]
    seg = np.array([x1 - x0, y1 - y0])
    seg_len2 = float(seg @ seg)
    if seg_len2 < 1e-12:
        return path_phis[i1]
    frac = float(np.clip((np.array(ball_xy) - [x0, y0]) @ seg / seg_len2, 0.0, 1.0))
    return path_phis[i0] + angle_diff_signed(path_phis[i1], path_phis[i0]) * frac


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--video", type=str, default=None,
                         help=f"Filename to save an MP4 recording under {VIDEO_DIR} "
                              "(an absolute path is used as-is; requires ffmpeg on PATH)")
    return parser.parse_args()


def resolve_video_path(video_arg):
    if not video_arg:
        return None
    video_path = video_arg if os.path.isabs(video_arg) else os.path.join(VIDEO_DIR, video_arg)
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    return video_path


def main():
    args = parse_args()
    seed = args.seed if args.seed is not None else random.randrange(2 ** 31)
    rng = random.Random(seed)
    print(f"course seed: {seed}")

    video_path = resolve_video_path(args.video)

    obstacles, path, path_phis_list, attempts = generate_solvable_course(rng)
    path_xs = np.array([w[0] for w in path])          # planned robot-body (chassis) path
    path_ys = np.array([w[1] for w in path])
    path_thetas = np.array([w[2] for w in path])
    path_phis = np.array(path_phis_list)
    # The ball/paddle's own target path: derived from the robot-body path by
    # the same arm offset Robot.set_pose uses (chassis = paddle + ARM_LENGTH*
    # unit(phi)), solved for the paddle side. This is what the ball is
    # actually aimed at and what phi is interpolated against live, since
    # those need a point close to where the ball really is, not the chassis.
    paddle_xs = path_xs - ARM_LENGTH * np.cos(path_phis)
    paddle_ys = path_ys - ARM_LENGTH * np.sin(path_phis)
    path_s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(path_xs), np.diff(path_ys)))])
    path_length = float(path_s[-1])
    print(f"generated a solvable {len(obstacles)}-obstacle course after {attempts} attempt(s); "
          f"path length {path_length:.2f}m vs {COURSE_LENGTH:.2f}m straight "
          f"(+{path_length - COURSE_LENGTH:.2f}m overhead); "
          f"max heading swing {math.degrees(np.max(np.abs(path_thetas))):.0f}deg")

    connect_gui(COURSE_LENGTH)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(DT)

    # Starts capturing before anything is spawned so the recording includes
    # the obstacle course and planned path, not just the ball/robot run
    # through it. Needs ffmpeg on PATH; pybullet shells out to it to encode
    # the MP4.
    video_log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_path) if video_path else None

    plane = p.loadURDF("plane.urdf")
    p.changeDynamics(plane, -1, restitution=FLOOR_E)

    spawn_obstacles(obstacles)
    # Yellow line is the planned path -- the robot body's own center, not
    # the ball (the ball instead tracks the derived paddle_xs/paddle_ys
    # offset from it by the arm).
    for i in range(len(path) - 1):
        p.addUserDebugLine([path_xs[i], path_ys[i], 0.01], [path_xs[i + 1], path_ys[i + 1], 0.01],
                            lineColorRGB=[1, 1, 0], lineWidth=2)

    # Course boundary: the full navigable rectangle the ball/robot pair may use.
    corners = [(0, -COURSE_HALF_WIDTH), (COURSE_LENGTH, -COURSE_HALF_WIDTH),
               (COURSE_LENGTH, COURSE_HALF_WIDTH), (0, COURSE_HALF_WIDTH)]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        p.addUserDebugLine([a[0], a[1], 0.01], [b[0], b[1], 0.01], lineColorRGB=[1, 1, 1], lineWidth=2)

    ball = spawn_ball()
    robot = Robot()
    bounce = BallisticModel()

    # --- Simulation loop -----------------------------------------------------
    push_timer = 0
    contact_cooldown = 0
    heading = 0.0
    target, path_idx = plan_next_contact(paddle_xs, paddle_ys, [0.0, 0.0], 0, 0.0, bounce)
    contact_log = []
    t = 0.0
    # Scaled to the planned path's actual arc length, not just straight-line
    # course length: heading swings mean the ball travels farther than
    # COURSE_LENGTH to cover it, and a fixed cap could cut off a valid run.
    max_steps = int(max(60.0, 2.0 * path_length / LANE_VX) / DT)

    for step in range(max_steps):
        p.stepSimulation()
        t += DT

        ball_pos, _ = p.getBasePositionAndOrientation(ball)
        ball_vel, _ = p.getBaseVelocity(ball)
        vz = ball_vel[2]

        # The paddle must coincide with the ball at contact anyway, so
        # there's no need to *predict* where it should be -- just make it
        # the ball's real current position every step (no integration, no
        # timing model, nothing that can drift or diverge from reality).
        # heading only changes at a contact (the ball is a free projectile
        # between them, so its direction of travel is genuinely fixed
        # until then).
        #
        # phi (the arm's offset angle) is looked up from the precomputed,
        # fully-verified sequence (plan_phi_sequence) -- interpolated by
        # the ball's actual position along its own target path
        # (paddle_xs/ys, the arm-offset projection of the planned
        # robot-body path), matching exactly what was validated. No live
        # obstacle check needed here at all.
        phi = phi_at_progress(paddle_xs, paddle_ys, path_phis, ball_pos[:2], path_idx)

        paddle_z = PADDLE_CENTER_Z - PUSH_DEPTH * (push_timer / PUSH_STEPS) if push_timer > 0 else PADDLE_CENTER_Z
        push_timer = max(push_timer - 1, 0)
        robot.set_pose(ball_pos[:2], phi, paddle_z)

        # No vz>0 gate: mirror_law_push is an identity in vz, and gating on
        # its sign can silently drop the contact if PyBullet's own
        # collision response zeroes vz before this code sees it.
        if robot.touches(ball) and contact_cooldown <= 0:
            v_new_z = mirror_law_push(vz)
            target, path_idx = plan_next_contact(paddle_xs, paddle_ys, ball_pos[:2], path_idx, t, bounce)
            heading = math.atan2(target.push_dir[1], target.push_dir[0])
            p.resetBaseVelocity(ball, linearVelocity=[*(LANE_VX * target.push_dir), v_new_z])
            contact_cooldown, push_timer = 20, PUSH_STEPS

            contact_log.append(ContactEvent(t, ball_pos[0], ball_pos[1], math.degrees(heading)))
            print(f"contact #{len(contact_log):3d}  t={t:6.2f}s  xy=({ball_pos[0]:.3f}, {ball_pos[1]:.3f})m  "
                  f"heading={math.degrees(heading):5.1f}deg -> next xy=({target.xy[0]:.3f}, {target.xy[1]:.3f})m "
                  f"@ t={target.t:.2f}s")
        else:
            contact_cooldown -= 1

        if ball_pos[0] >= COURSE_LENGTH:
            print("\nCourse complete.")
            break

        time.sleep(DT)

    if video_log_id is not None:
        p.stopStateLogging(video_log_id)
        print(f"Video saved to {video_path}")

    p.disconnect()

    print(f"Contacts recorded: {len(contact_log)}")
    if contact_log:
        headings = [c.heading_deg for c in contact_log]
        print(f"Heading ranged {min(headings):.0f}deg to {max(headings):.0f}deg during the run")


if __name__ == "__main__":
    main()
