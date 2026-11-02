"""
Court Vision — Objective 3 scaffold, v3
Obstacle course: a randomly generated course of 6-9 common obstacles
(crates, pillars, walls), each randomly placed and oriented, about half of
them oscillating back and forth on a known schedule (Objective 4) rather
than sitting still. The ball and robot move together like a player
dribbling through a crowd: the planned
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
  2. planning.hybrid_astar(): a state-lattice search over (x, y, heading,
     time) -- not just (x, y) -- since the robot body needs position,
     orientation, *and* a known arrival time to route through a gap only
     wide enough at an angle, or past an obstacle that's only there
     sometimes (a minority of obstacles oscillate on a known schedule,
     Obstacle.position_at). Each step is a short forward arc at one of a
     few discrete turning rates, advancing time at an assumed constant
     speed. Both the robot body's own point and the existence of *some*
     wide-enough arm angle (clearing the ball/paddle end) are checked for
     clearance, against each obstacle's position at that exact time, at
     every step.
  3. planning.plan_phi_sequence(): a second search, over the already-found
     path, that picks a specific arm angle at every point and verifies
     every transition between consecutive points is fully collision-free
     -- not just that some angle works at each point in isolation, which
     doesn't guarantee the ball/paddle can actually get from one point's
     safe angle to the next's. This is what makes the run free of live
     emergency corrections, moving obstacles included: the whole arm
     trajectory is proven traversable before the simulation ever starts,
     not discovered (and occasionally failed to find) reactively frame by
     frame.
  If a random course turns out unsolvable at either stage, generation
  retries with a fresh course rather than failing.

  main() below then ties it together: spawns the pybullet world
  (obstacles, planned-path debug line, ball, robot), and runs the
  dribble-and-push control loop (physics.py, robot.py) frame by frame.

Run:
    pip install numpy pybullet
    python simulation.py [--seed N] [--planner dwa|dstar|hybrid|offline] [--speed 3] [--video out.mp4]

    --video saves under term_project/video_runs/ (created if missing) --
    pass an absolute path instead to save elsewhere. Requires ffmpeg on
    PATH (pybullet shells out to it to encode the MP4) and records the
    whole GUI window from course spawn to finish.
"""

import argparse
import hashlib
import math
import os
import pickle
import random
import time

import numpy as np
import pybullet as p
import pybullet_data

from config import (
    ARM_LENGTH, BALL_MASS, BALL_RADIUS, BALL_SPEED_MAX, CONTACT_Z, COURSE_HALF_WIDTH, COURSE_LENGTH,
    DT, FLOOR_E, FLOOR_Z, G, PADDLE_CENTER_Z, ROBOT_RADIUS,
)
from display import connect_gui
from evaluation import plan_constraints, summarize_redirects
from live_planning import LivePlanner
from obstacles import RoamingField, move_obstacles, obstacle_blocks, spawn_obstacles
from physics import BallisticModel, ContactEvent, ContactTarget, paddle_impact, spawn_ball
from planning import angle_diff_signed, ball_in_bounds, generate_solvable_course, paddle_xy, plan_course
from robot import Robot

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(HERE, "video_runs")
COURSE_CACHE_DIR = os.path.join(HERE, ".course_cache")
PATH_DRAW_HZ = 2   # live route redraws per wall-clock second


def draw_path_points(xy, replace_id=-1):
    """Draws a path as one batch of closely spaced points -- a single
    debug-draw call however long the path. Each debug-draw call waits on
    the GUI renderer (~20ms under WSL's software OpenGL), so drawing a
    path as one line per segment stalled the simulation for over a second
    every redraw."""
    pts = []
    for (x0, y0), (x1, y1) in zip(xy, xy[1:]):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / 0.02))
        pts += [[x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n, 0.01] for k in range(n)]
    if xy:
        pts.append([xy[-1][0], xy[-1][1], 0.01])
    if replace_id >= 0:
        p.removeUserDebugItem(replace_id)
    if not pts:
        return -1
    return p.addUserDebugPoints(pts, [[1, 1, 0]] * len(pts), pointSize=3)


def cached_solvable_course(seed, motion, ignore_moving=False):
    """generate_solvable_course for this seed, saved to disk the first
    time so relaunching the same seed skips the 1-10s of course generation
    and planning. Keyed on the source of every module that shapes the
    result, so any change to the planner, the obstacles, or the constants
    regenerates rather than reusing a stale course."""
    src = hashlib.sha256()
    for name in ("planning.py", "obstacles.py", "config.py"):
        with open(os.path.join(HERE, name), "rb") as f:
            src.update(f.read())
    # Course parameters an experiment may override at runtime (experiments.py)
    # without editing any file, so they have to be part of the key too.
    import obstacles as obstacle_module
    src.update(repr((obstacle_module.N_OBSTACLES_RANGE, obstacle_module.MIN_OBSTACLE_GAP,
                     obstacle_module.ROAM_SPEED_RANGE, obstacle_module.DYNAMIC_OBSTACLE_FRACTION,
                     obstacle_module.ROAM_TURN_RATE)).encode())
    tag = "static" if ignore_moving else "full"
    path = os.path.join(COURSE_CACHE_DIR, f"{seed}_{motion}_{tag}_{src.hexdigest()[:12]}.pkl")
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, pickle.UnpicklingError, EOFError):
        pass
    result = generate_solvable_course(random.Random(seed), motion=motion, ignore_moving=ignore_moving)
    os.makedirs(COURSE_CACHE_DIR, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return result

PUSH_STEPS = 12   # length of the drawn push stroke; its depth follows the paddle's real speed


class PlannedSchedule:
    """The plan as a function of time: robot-body position and arm angle
    linearly interpolated between consecutive path points by their planned
    times (path_ts), the ball/paddle position derived from them by the arm
    offset -- exactly the (x, y, phi, t) interpolation
    planning.phi_transition_clear verified, so any instant this returns
    was proven clear of every obstacle at that instant, moving ones
    included. Times outside the plan clamp to its endpoints."""

    def __init__(self, path_xs, path_ys, path_phis, path_ts):
        self.xs, self.ys, self.ts = path_xs, path_ys, path_ts
        # unwrap takes the same shortest way around between consecutive
        # angles that angle_diff_signed does in the planner.
        self.phis = np.unwrap(path_phis)
        self.t_end = float(path_ts[-1])

    def phi_at(self, t):
        return float(np.interp(t, self.ts, self.phis))

    def paddle_at(self, t):
        x, y = np.interp(t, self.ts, self.xs), np.interp(t, self.ts, self.ys)
        return np.array(paddle_xy(x, y, self.phi_at(t)))


def plan_next_contact(schedule, ball_xy, t_now, bounce: BallisticModel):
    """Aims the ball at where the plan says the paddle should be at the
    *next* contact's time, pushing at whatever horizontal speed covers
    that distance in one flight (capped at BALL_SPEED_MAX) -- not at a
    fixed LANE_VX toward the next waypoint. That earlier waypoint-follower
    tracked only *where* along the path the ball was, never *when*: floor-
    bounce friction, arm swings, and the start offset all quietly
    accumulated as lag (measured: 3-19s behind schedule by the end of a
    run), harmless against static obstacles but fatal against moving ones,
    which the planner only proved clear at the planned time. Re-deriving
    both the target and the speed from the ball's real position and the
    real clock at every contact means any lag is caught up within a
    bounce or two rather than compounding."""
    t_next = t_now + bounce.flight_time
    target_xy = schedule.paddle_at(t_next)
    delta = target_xy - np.array(ball_xy)
    dist = np.linalg.norm(delta)
    push_dir = delta / dist if dist > 1e-9 else np.array([1.0, 0.0])
    speed = min(dist / bounce.flight_time, BALL_SPEED_MAX)
    return ContactTarget(xy=target_xy, t=t_next, push_dir=push_dir), speed


class NoPlan(Exception):
    """The oracle couldn't plan the live planners' course for this seed."""


class OfflineController:
    """Planner proven in advance (planning.py), followed on its own clock.

    oracle=True plans, with full knowledge of the obstacles' motion, on the
    very course the live planners get for this seed (raising NoPlan if it
    can't) -- an upper bound to compare them against. Otherwise it draws
    courses until one is fully solvable."""

    def __init__(self, seed, motion, oracle=False):
        if oracle:
            self.obstacles = cached_solvable_course(seed, motion, ignore_moving=True)[0]
            path, path_phis_list, _reason = plan_course(self.obstacles)
            if path is None:
                raise NoPlan(seed)
            attempts = 1
        else:
            self.obstacles, path, path_phis_list, attempts = cached_solvable_course(seed, motion)
        path_xs = np.array([w[0] for w in path])          # planned robot-body (chassis) path
        path_ys = np.array([w[1] for w in path])
        path_thetas = np.array([w[2] for w in path])
        path_ts = np.array([w[3] for w in path])          # assumed wall-clock time at each path point
        path_phis = np.array(path_phis_list)
        # The ball/paddle's own target path: derived from the robot-body path by
        # the same arm offset Robot.set_pose uses (chassis = paddle + ARM_LENGTH*
        # unit(phi)), solved for the paddle side. Live, the ball is aimed along
        # this by time (PlannedSchedule), not just by position.
        paddle_xs = path_xs - ARM_LENGTH * np.cos(path_phis)
        paddle_ys = path_ys - ARM_LENGTH * np.sin(path_phis)
        self.schedule = PlannedSchedule(path_xs, path_ys, path_phis, path_ts)
        self.path_xs, self.path_ys = path_xs, path_ys
        path_length = float(np.sum(np.hypot(np.diff(path_xs), np.diff(path_ys))))
        n_moving = sum(1 for o in self.obstacles if o.moving)
        n_waits = int(np.sum((np.diff(path_xs) == 0) & (np.diff(path_ys) == 0)))
        print(f"generated a solvable {len(self.obstacles)}-obstacle course ({n_moving} moving) after "
              f"{attempts} attempt(s); path length {path_length:.2f}m vs {COURSE_LENGTH:.2f}m straight "
              f"(+{path_length - COURSE_LENGTH:.2f}m overhead), {n_waits} wait step(s); "
              f"max heading swing {math.degrees(np.max(np.abs(path_thetas))):.0f}deg")
        redirects = summarize_redirects(self.obstacles, paddle_xs, paddle_ys)
        print(f"redirect events (Obj. 4): {redirects['n_redirects']}/{redirects['n_points']} path points "
              f"({redirects['redirect_fraction']:.0%}) had a blocked nominal (straight-ahead) landing spot; "
              f"mean shift {redirects['mean_shift']:.2f}m, max shift {redirects['max_shift']:.2f}m")
        moves = [(math.dist(a[:2], b[:2]), b[3] - a[3]) for a, b in zip(path, path[1:]) if a[:2] != b[:2]]
        speeds = [d / dt for d, dt in moves]
        print(f"travel speed along the plan: {min(speeds):.2f}-{max(speeds):.2f} m/s "
              f"(mean {sum(d for d, _ in moves) / sum(dt for _, dt in moves):.2f}), slower where the way is tighter")
        self.constraints = plan_constraints(self.obstacles, path, path_phis_list)
        c = self.constraints
        print(f"constraints: nominal landing spots both blocked at {c['redirect_forced_frac']:.0%} of path points; "
              f"binding on transitions -- base turn rate {c['turn_rate_binding_frac']:.0%}, "
              f"base speed {c['base_speed_binding_frac']:.0%}, ball speed {c['ball_speed_binding_frac']:.0%}")
        self.t_limit = self.schedule.t_end + 30.0

    def start_ball_xy(self):
        return self.schedule.paddle_at(0.0)

    def draw(self):
        # Yellow line is the planned path -- the robot body's own center, not
        # the ball (the ball instead tracks the paddle offset from it by the arm).
        draw_path_points(list(zip(self.path_xs, self.path_ys)))

    def phi_at(self, t):
        return self.schedule.phi_at(t)

    def on_contact(self, t, ball_xy, bounce):
        target, speed = plan_next_contact(self.schedule, ball_xy, t, bounce)
        return target, speed

    def finished(self, t, ball_xy, body_xy):
        # The plan's last point is past COURSE_LENGTH for the robot body,
        # but not necessarily for the ball an arm's length away -- so
        # reaching the end of the schedule also counts.
        at_end = t >= self.schedule.t_end and np.linalg.norm(np.array(ball_xy) - self.schedule.paddle_at(t)) < 0.1
        return ball_xy[0] >= COURSE_LENGTH or at_end

    def summary(self):
        return dict(self.constraints)


class LiveController:
    """Replanning at every contact from sensed obstacle positions
    (live_planning.py, with one of its search strategies); knows nothing of
    where a roaming obstacle is going."""

    START_PHI = math.pi / 2   # ball starts beside the robot body, which starts at the origin

    def __init__(self, seed, motion, headless, search):
        # Courses whose static obstacles leave room for the body *and* the
        # arm (the offline planner, run with the moving obstacles left out;
        # its plan is thrown away). Accepting any course with a path on the
        # coarse grid let in gaps too tight for the robot with its ball and
        # arm beside it -- live runs failed some courses that had no moving
        # obstacles at all. Whether the moving obstacles then let it
        # through is exactly what a live run finds out.
        self.obstacles, _path, _phis, attempt = cached_solvable_course(seed, motion, ignore_moving=True)
        self.planner = LivePlanner(self.obstacles, (0.0, 0.0), search)
        n_moving = sum(1 for o in self.obstacles if o.moving)
        print(f"generated a {len(self.obstacles)}-obstacle course ({n_moving} moving) after {attempt} attempt(s); "
              f"live planner: {search}")
        self.headless = headless
        self.phi_from = self.phi_to = self.START_PHI
        self.t_from, self.t_to = 0.0, 1.0
        self.path_draw_id = -1
        self.next_path_draw = 0.0
        self.replan_s = []
        self.t_limit = 120.0

    def start_ball_xy(self):
        return np.array(paddle_xy(0.0, 0.0, self.START_PHI))

    def draw(self):
        pass   # the path is redrawn at every replan instead

    def phi_at(self, t):
        f = min(max((t - self.t_from) / (self.t_to - self.t_from), 0.0), 1.0)
        return self.phi_from + angle_diff_signed(self.phi_to, self.phi_from) * f

    def on_contact(self, t, ball_xy, bounce):
        t0 = time.perf_counter()
        plan = self.planner.plan_contact(t, ball_xy, self.phi_at(t), bounce.flight_time)
        self.replan_s.append(time.perf_counter() - t0)
        self.phi_from, self.phi_to = self.phi_at(t), plan.phi
        self.t_from, self.t_to = t, t + bounce.flight_time
        if not self.headless and time.perf_counter() >= self.next_path_draw:
            self.path_draw_id = draw_path_points(plan.path, self.path_draw_id)
            self.next_path_draw = time.perf_counter() + 1.0 / PATH_DRAW_HZ
        delta = plan.paddle_xy - np.asarray(ball_xy)
        dist = np.linalg.norm(delta)
        push_dir = delta / dist if dist > 1e-9 else np.array([1.0, 0.0])
        return ContactTarget(xy=plan.paddle_xy, t=self.t_to, push_dir=push_dir), plan.speed

    def finished(self, t, ball_xy, body_xy):
        return body_xy[0] >= COURSE_LENGTH

    def summary(self):
        r = self.replan_s
        return {"replans": self.planner.replans, "no_path_contacts": self.planner.no_path_contacts,
                "replan_ms_mean": 1000 * sum(r) / len(r), "replan_ms_max": 1000 * max(r),
                "expansions": self.planner.dstar.expansions,
                "cruise_mean": float(np.mean([c for _, c in self.planner.cruise_log])) if self.planner.cruise_log else float("nan"),
                **{f"binding_{k}": v / self.planner.replans for k, v in self.planner.binding.items()}}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--planner", choices=["dwa", "dstar", "hybrid", "offline", "oracle"], default="dwa",
                        help="live planners, replanning every contact from sensed obstacles: dwa (D* Lite "
                             "route + dynamic-window local step; default), dstar (D* Lite alone), hybrid "
                             "(hybrid A* rerun from scratch). offline: the whole run planned and proven in "
                             "advance from the obstacles' known motion. oracle: offline, but on the same "
                             "course the live planners get for this seed")
    parser.add_argument("--motion", choices=["roam", "oscillate"], default=None,
                        help="how moving obstacles move: roam (random direction, bouncing; default for "
                             "the live planners) or oscillate (known sinusoid; default for --planner offline)")
    parser.add_argument("--speed", type=float, default=3.0,
                        help="GUI playback speed as a multiple of real time (default 3; 1 = real time, "
                             "0 = as fast as the machine can go)")
    parser.add_argument("--headless", action="store_true",
                        help="no GUI and no real-time pacing -- for batch evaluation")
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


def run(seed, planner="dwa", motion=None, headless=False, video_path=None, verbose=True, playback_speed=3.0):
    """One full run; returns a dict of outcome metrics (complete, time,
    collision counts, and the planner's own stats)."""
    live = planner not in ("offline", "oracle")
    motion = motion or ("oscillate" if planner == "offline" else "roam")
    print(f"course seed: {seed}  planner: {planner}  obstacle motion: {motion}")
    if live:
        controller = LiveController(seed, motion, headless, planner)
    else:
        try:
            controller = OfflineController(seed, motion, oracle=planner == "oracle")
        except NoPlan:
            print("the oracle found no plan for this course")
            return {"seed": seed, "planner": planner, "motion": motion, "complete": False, "no_plan": True}
    obstacles = controller.obstacles

    if headless:
        p.connect(p.DIRECT)
    else:
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

    obstacle_body_ids = spawn_obstacles(obstacles)
    # Roaming obstacles are stepped live from here on, so they can bounce
    # off the robot instead of passing through it.
    field = RoamingField(obstacles)
    if not headless:
        controller.draw()
        # Course boundary: the full navigable rectangle the ball/robot pair may use.
        corners = [(0, -COURSE_HALF_WIDTH), (COURSE_LENGTH, -COURSE_HALF_WIDTH),
                   (COURSE_LENGTH, COURSE_HALF_WIDTH), (0, COURSE_HALF_WIDTH)]
        for a, b in zip(corners, corners[1:] + corners[:1]):
            p.addUserDebugLine([a[0], a[1], 0.01], [b[0], b[1], 0.01], lineColorRGB=[1, 1, 1], lineWidth=2)

    robot = Robot()
    bounce = BallisticModel()
    # The ball starts where the plan starts it (an arm's length from the
    # robot body's start point, at the planned first arm angle) and already
    # heading for its first planned contact -- not at the origin, which
    # put it most of an arm's length behind schedule before the run began.
    start_xy = controller.start_ball_xy()
    target, speed = controller.on_contact(0.0, start_xy, bounce)
    ball = spawn_ball(start_xy, speed * target.push_dir)
    # The paddle's contact with the ball is modeled, not simulated
    # (physics.paddle_impact): PyBullet's own response to the two bodies
    # touching would act on the ball first and confuse the energy
    # accounting, so they don't collide. Obstacles and the floor still do.
    p.setCollisionFilterPair(robot.paddle_id, ball, -1, -1, 0)

    # --- Simulation loop -----------------------------------------------------
    push_timer = 0
    contact_cooldown = 0
    heading = 0.0
    contact_log = []
    t = 0.0
    complete = False
    outcome = "timeout"
    # Ground truth, independent of what either planner believed: real
    # body/ball positions against every obstacle's real position at the
    # real time, with no planning margin. A "hit" is one continuous stretch
    # of overlap with one obstacle, however many steps it lasts.
    overlap_steps = {"body": 0, "ball": 0}
    hits = {"body": 0, "ball": 0}
    touching = set()
    # Ball energy ledger (J). Every step's change in the ball's mechanical
    # energy is attributed: paddle work and impact loss at contacts (from
    # the impact model), floor loss while touching the floor (measured),
    # and PyBullet's semi-implicit Euler integrator, which loses exactly
    # (1/2) m g^2 dt^2 per step under gravity. Anything else -- obstacle
    # hits, or a bookkeeping gap -- is "other", which is what shows whether
    # energy is actually being accounted for.
    energy = {"paddle_work": 0.0, "impact_loss": 0.0, "floor_loss": 0.0, "integrator_loss": 0.0, "other": 0.0}
    integrator_loss_per_step = 0.5 * BALL_MASS * G ** 2 * DT ** 2
    body_path_len, body_prev = 0.0, None
    ball_path_len, ball_prev = 0.0, None   # horizontal; the chassis also swings around the ball as phi turns

    spin_inertia = 0.4 * BALL_MASS * BALL_RADIUS ** 2   # solid sphere

    def ball_energy(pos, vel, ang=(0.0, 0.0, 0.0)):
        # Spin counts too: floor friction converts some speed into spin.
        return (0.5 * BALL_MASS * (vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
                + 0.5 * spin_inertia * (ang[0] ** 2 + ang[1] ** 2 + ang[2] ** 2) + BALL_MASS * G * pos[2])

    e_start = e_prev = ball_energy(p.getBasePositionAndOrientation(ball)[0], *p.getBaseVelocity(ball))
    paddle_normal, push_depth = None, 0.0

    # Playback is paced against the wall clock (sleep only while ahead of
    # t/playback_speed), not by a fixed sleep per step -- a fixed time.sleep(DT) on
    # top of each step's own physics, planning, and rendering time always
    # ran slower than real time.
    wall_start = time.perf_counter()
    for step in range(int(controller.t_limit / DT)):
        p.stepSimulation()
        t += DT
        # Moving obstacles' real bodies follow their real motion (the same
        # Obstacle.position_at the offline planner reads in advance and the
        # live planner only ever senses at the current instant).
        field.advance(t, body_prev if body_prev is not None else robot.base_xy)
        move_obstacles(obstacles, obstacle_body_ids, t)

        ball_pos, _ = p.getBasePositionAndOrientation(ball)
        ball_vel, ball_ang = p.getBaseVelocity(ball)
        vz = ball_vel[2]
        e_now = ball_energy(ball_pos, ball_vel, ball_ang)
        # Floor contact can span more than one step, so every step the
        # ball spends touching the floor (not just the one where vz flips)
        # is floor loss.
        if p.getContactPoints(ball, plane):
            energy["floor_loss"] += e_prev - e_now
        else:
            energy["integrator_loss"] += integrator_loss_per_step
            energy["other"] += e_now - e_prev + integrator_loss_per_step
        e_prev = e_now

        # The paddle must coincide with the ball at contact anyway, so
        # there's no need to *predict* where it should be -- just make it
        # the ball's real current position every step (no integration, no
        # timing model, nothing that can drift or diverge from reality).
        # phi (the arm's offset angle) comes from the controller: the
        # precomputed, fully-verified sequence (offline), or a smooth turn
        # toward the angle chosen at the last contact (live).
        phi = controller.phi_at(t)

        # Push stroke: the paddle follows through along its tilted face at
        # the speed the impact needed, then returns flat.
        pushing = push_timer > 0
        paddle_z = PADDLE_CENTER_Z - push_depth * (push_timer / PUSH_STEPS) if pushing else PADDLE_CENTER_Z
        push_timer = max(push_timer - 1, 0)
        body_xy = robot.set_pose(ball_pos[:2], phi, paddle_z, paddle_normal if pushing else None)
        if body_prev is not None:
            body_path_len += math.dist(body_xy, body_prev)
            ball_path_len += math.dist(ball_pos[:2], ball_prev)
        body_prev, ball_prev = body_xy, ball_pos[:2]

        for part, (x, y), r in (("body", body_xy, ROBOT_RADIUS), ("ball", ball_pos[:2], BALL_RADIUS)):
            for i, o in enumerate(obstacles):
                if obstacle_blocks(o, x, y, r, t):
                    overlap_steps[part] += 1
                    if (part, i) not in touching:
                        hits[part] += 1
                        touching.add((part, i))
                else:
                    touching.discard((part, i))

        # Contact: the rising ball reaches the paddle's face.
        if ball_pos[2] >= CONTACT_Z and vz > 0 and contact_cooldown <= 0:
            target, speed = controller.on_contact(t, ball_pos[:2], bounce)
            heading = math.atan2(target.push_dir[1], target.push_dir[0])
            imp = paddle_impact(ball_vel, speed * target.push_dir)
            p.resetBaseVelocity(ball, linearVelocity=list(imp.v_out))
            energy["paddle_work"] += imp.work
            energy["impact_loss"] += imp.loss
            e_prev = ball_energy(ball_pos, imp.v_out, ball_ang)
            paddle_normal, push_depth = imp.normal, imp.paddle_speed * PUSH_STEPS * DT
            contact_cooldown, push_timer = 20, PUSH_STEPS

            contact_log.append(ContactEvent(t, ball_pos[0], ball_pos[1], math.degrees(heading), imp))
            if verbose:
                print(f"contact #{len(contact_log):3d}  t={t:6.2f}s  xy=({ball_pos[0]:.3f}, {ball_pos[1]:.3f})m  "
                      f"heading={math.degrees(heading):5.1f}deg -> next xy=({target.xy[0]:.3f}, {target.xy[1]:.3f})m "
                      f"@ t={target.t:.2f}s")
        else:
            contact_cooldown -= 1

        if controller.finished(t, ball_pos[:2], body_xy):
            complete, outcome = True, "finished"
            print("\nCourse complete.")
            break
        # A run is over once the dribble is: the ball has left the course,
        # or a few bounce cycles have passed with no contact (knocked away,
        # or stopped short of the paddle). Either way nothing can bring it
        # back, and waiting out the time limit would hide what happened.
        if not ball_in_bounds(*ball_pos[:2]) and ball_pos[0] < COURSE_LENGTH:
            outcome = "out_of_bounds"
            print("\nBall out of bounds.")
            break
        last_contact_t = contact_log[-1].t if contact_log else 0.0
        if t - last_contact_t > 4 * bounce.flight_time:
            outcome = "dribble_lost"
            print("\nDribble lost.")
            break

        if not headless and playback_speed > 0:
            ahead = wall_start + t / playback_speed - time.perf_counter()
            if ahead > 0:
                time.sleep(ahead)

    if video_log_id is not None:
        p.stopStateLogging(video_log_id)
        print(f"Video saved to {video_path}")

    p.disconnect()

    result = {"seed": seed, "planner": planner, "motion": motion, "complete": complete, "outcome": outcome, "time_s": t,
              "contacts": len(contact_log), "moving": sum(1 for o in obstacles if o.moving),
              "obstacles": len(obstacles), "body_path_len": body_path_len, "ball_path_len": ball_path_len,
              "body_hits": hits["body"], "ball_hits": hits["ball"],
              "body_overlap_s": overlap_steps["body"] * DT, "ball_overlap_s": overlap_steps["ball"] * DT,
              "energy_change": e_prev - e_start, **energy,
              "paddle_tilt_max_deg": max((math.degrees(c.impact.tilt) for c in contact_log), default=0.0),
              "paddle_speed_max": max((c.impact.paddle_speed for c in contact_log), default=0.0),
              "tilt_limited_contacts": sum(c.impact.tilt_limited for c in contact_log),
              **controller.summary()}
    print(f"{'complete' if complete else 'DID NOT FINISH (' + outcome.replace('_', ' ') + ')'} in {t:.1f}s, "
          f"{len(contact_log)} contacts; "
          f"collisions: body {hits['body']} ({result['body_overlap_s']:.2f}s overlapping), "
          f"ball {hits['ball']} ({result['ball_overlap_s']:.2f}s overlapping)")
    print(f"ball energy: change {result['energy_change']:+.2f}J = paddle work {energy['paddle_work']:.1f}J "
          f"- impact loss {energy['impact_loss']:.1f}J - floor loss {energy['floor_loss']:.1f}J "
          f"- integrator loss {energy['integrator_loss']:.2f}J + other {energy['other']:+.2f}J; paddle tilt up to {result['paddle_tilt_max_deg']:.0f}deg, "
          f"speed up to {result['paddle_speed_max']:.2f}m/s")
    if live:
        print(f"live planner: {result['replans']} replans, mean {result['replan_ms_mean']:.1f}ms / "
              f"max {result['replan_ms_max']:.1f}ms each, {result['no_path_contacts']} contact(s) with no path "
              f"(dribbled in place)")
        print("what ruled out the preferred arm angle, by share of contacts: " + ", ".join(
            f"{k[len('binding_'):]} {v:.0%}" for k, v in sorted(result.items()) if k.startswith("binding_")))
    return result


def main():
    args = parse_args()
    seed = args.seed if args.seed is not None else random.randrange(2 ** 31)
    run(seed, args.planner, args.motion, args.headless, resolve_video_path(args.video), playback_speed=args.speed)


if __name__ == "__main__":
    main()
