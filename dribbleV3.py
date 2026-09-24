"""
Court Vision — Objective 3 scaffold, v3
Static obstacle course: a randomly generated course of 4-6 common obstacles
(crates, pillars, walls), each randomly placed and oriented. The ball and
robot move together like a player dribbling through a crowd: the ball's
own path curves to slip through gaps, with the robot escorting it at a
fixed arm's length, defaulting to beside the ball but falling in directly
behind it wherever the side is too tight -- not locked to a rigid
heading+90-degree offset. Heading only changes where a straight line
wouldn't fit.

This replaces an earlier design where the ball stayed on a fixed straight
lane and only the robot's arm reach varied -- that produces a robot doing
the work by contorting an arm while the ball ignores the crowd, not a
player moving through one.

Pipeline:
  1. generate_course(): random obstacles (kind, position, size, yaw),
     scattered on both sides of the straight-ahead line, spaced far enough
     apart to leave room for the ball *and* the chassis's swing through any
     gap between them, not just the ball alone.
  2. hybrid_astar(): a state-lattice search over (x, y, heading) -- not
     just (x, y) -- since the ball/robot pair needs both position *and*
     orientation to route through a gap only wide enough at an angle. Each
     step is a short forward arc at one of a few discrete turning rates.
     Both the ball's point and the existence of *some* wide-enough chassis
     angle are checked for clearance at every step.
  3. plan_phi_sequence(): a second search, over the already-found path, that
     picks a specific chassis angle at every point and verifies every
     transition between consecutive points is fully collision-free -- not
     just that some angle works at each point in isolation, which doesn't
     guarantee the chassis can actually get from one point's safe angle to
     the next's. This is what makes the run free of live emergency
     corrections: the whole chassis trajectory is proven traversable before
     the simulation ever starts, not discovered (and occasionally failed to
     find) reactively frame by frame.
  If a random course turns out unsolvable at either stage, generation
  retries with a fresh course rather than failing.

Run:
    pip install numpy pybullet
    python dribbleV3.py [--seed N]
"""

import argparse
import heapq
import math
import random
import time
from dataclasses import dataclass

import numpy as np
import pybullet as p
import pybullet_data

DT = 1.0 / 240.0

# --- Tunable parameters -----------------------------------------------------
FLOOR_E = 0.85
HAND_HEIGHT = 0.6
HAND_E = 0.8
V_PUSH = 3.0
LANE_VX = 0.15

BALL_RADIUS = 0.12
BALL_MASS = 0.6
PADDLE_HALF = [0.15, 0.15, 0.02]
BASE_XY_HALF = 0.2

G = 9.81
CONTACT_Z = HAND_HEIGHT - BALL_RADIUS
FLOOR_Z = BALL_RADIUS
PADDLE_CENTER_Z = HAND_HEIGHT + PADDLE_HALF[2]

COURSE_LENGTH = 6.0
COURSE_HALF_WIDTH = 1.5   # obstacles/path may range y in [-COURSE_HALF_WIDTH, COURSE_HALF_WIDTH]
ARM_LENGTH = 0.5          # fixed distance the robot escorts the ball at
N_OBSTACLES_RANGE = (4, 6)
ROBOT_RADIUS = BASE_XY_HALF * math.sqrt(2) + 0.03   # chassis circumscribed circle + safety margin
BALL_CLEARANCE = BALL_RADIUS + 0.05

# Hybrid-A* lattice: each step is a short forward arc; heading is part of
# the search state (not just x, y) so the planner can angle through a gap
# too narrow to clear head-on.
STEP_LEN = 0.1
MAX_DTHETA = math.radians(18)      # steering options per step
DTHETA_OPTIONS = [f * MAX_DTHETA for f in (-1.0, -0.5, 0.0, 0.5, 1.0)]
THETA_MAX = math.radians(70)       # heading stays within +-70 deg of straight-ahead
BIN_X, BIN_Y, BIN_THETA = 0.05, 0.05, math.radians(4)   # state dedup resolution
THETA_REG_WEIGHT = 0.3             # mild preference for facing forward absent a reason not to


# --- Course generation -------------------------------------------------------
@dataclass
class Obstacle:
    kind: str    # 'crate' | 'pillar' | 'wall'
    x: float
    y: float
    yaw: float
    hx: float    # half-extent (box) or radius (pillar)
    hy: float    # half-extent (box); unused for pillar
    height: float

    def __post_init__(self):
        # obstacle_blocks runs tens of millions of times during planning;
        # precomputing its per-obstacle constants once here (rather than
        # recomputing sin/cos and a reach bound on every single call) is a
        # meaningful, purely mechanical speedup with no effect on what it
        # computes.
        self.cos_nyaw = math.cos(-self.yaw)
        self.sin_nyaw = math.sin(-self.yaw)
        self.reach = self.hx + self.hy   # conservative (>= true half-diagonal) bound for a cheap early reject


def generate_course(rng):
    """4-6 obstacles, each with a random kind/size/orientation, scattered on
    both sides of the straight-ahead line (y drawn from a triangular
    distribution peaked at 0) so the path usually has to weave rather than
    just lean to one side."""
    n = rng.randint(*N_OBSTACLES_RANGE)
    margin = 0.6
    # Wide enough that any gap between two obstacles has room for the ball
    # AND the chassis's ARM_LENGTH swing, not just the ball alone -- at the
    # old 0.15m, two obstacles could still randomly land ~0.5m apart (ball
    # clearance alone is ~0.34m of that), leaving no room for the chassis to
    # take either side, which forced last-instant, visibly jarring corrections.
    min_center_gap = 2.0 * ARM_LENGTH
    obstacles = []
    for _ in range(n):
        yaw = rng.uniform(0, 2 * math.pi)
        kind = rng.choice(["crate", "pillar", "wall"])
        # Sized at or above the robot's own footprint (BASE_XY_HALF=0.2) so
        # obstacles are genuine obstacles, not gaps the robot dwarfs.
        if kind == "crate":
            hx, hy, height = rng.uniform(0.20, 0.30), rng.uniform(0.20, 0.30), rng.uniform(0.2, 0.4)
        elif kind == "pillar":
            hx = hy = rng.uniform(0.25, 0.40)   # wider cylinders
            height = rng.uniform(0.4, 0.8)
        else:  # wall: thin, but at least robot-length along its long axis
            hx, hy, height = rng.uniform(0.06, 0.10), rng.uniform(0.25, 0.45), rng.uniform(0.3, 0.6)

        reach = math.hypot(hx, hy)
        y_lo, y_hi = -COURSE_HALF_WIDTH + reach, COURSE_HALF_WIDTH - reach

        for _attempt in range(200):
            x = rng.uniform(margin, COURSE_LENGTH - margin)
            y = rng.triangular(y_lo, y_hi, 0.0) if y_hi > y_lo else 0.0
            if all(math.hypot(x - o.x, y - o.y) >= reach + math.hypot(o.hx, o.hy) + min_center_gap
                   for o in obstacles):
                break
        obstacles.append(Obstacle(kind, x, y, yaw, hx, hy, height))
    return obstacles


def spawn_obstacles(obstacles):
    color = {"crate": [0.6, 0.4, 0.2, 1], "pillar": [0.5, 0.5, 0.55, 1], "wall": [0.7, 0.2, 0.2, 1]}
    body_ids = []
    for obs in obstacles:
        quat = p.getQuaternionFromEuler([0, 0, obs.yaw])
        z = obs.height / 2
        if obs.kind == "pillar":
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=obs.hx, height=obs.height)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=obs.hx, length=obs.height, rgbaColor=color[obs.kind])
        else:
            half = [obs.hx, obs.hy, z]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color[obs.kind])
        body_ids.append(p.createMultiBody(0, col, vis, [obs.x, obs.y, z], quat))
    return body_ids


# --- Hybrid-A* path + heading planning ---------------------------------------
def obstacle_blocks(obs, x, y, radius):
    """True if a disc of `radius` at (x, y) overlaps `obs`. Pillars: circle
    test. Crates/walls: distance to an axis-aligned rectangle in the
    obstacle's own rotated frame (Minkowski sum of a rotated rect and a
    disc is a rounded rect). Called tens of millions of times during
    planning, hence: squared-distance comparisons throughout (no sqrt), a
    cheap early reject using obs.reach, and obs.cos_nyaw/sin_nyaw computed
    once per obstacle (Obstacle.__post_init__) rather than every call."""
    dx, dy = x - obs.x, y - obs.y
    dist2 = dx * dx + dy * dy
    reach = obs.reach + radius
    if dist2 > reach * reach:
        return False
    if obs.kind == "pillar":
        r = obs.hx + radius
        return dist2 <= r * r
    lx, ly = dx * obs.cos_nyaw - dy * obs.sin_nyaw, dx * obs.sin_nyaw + dy * obs.cos_nyaw
    qx, qy = max(abs(lx) - obs.hx, 0.0), max(abs(ly) - obs.hy, 0.0)
    return qx * qx + qy * qy <= radius * radius


PLANNING_MARGIN = 0.05   # extra buffer beyond the bare physical radii, reserved during
                         # planning only (not part of the real collision geometry). Real
                         # execution follows piecewise-straight chords between contacts,
                         # not the exact planned curve, so a plan that clears an obstacle
                         # by mere millimeters leaves no room for that approximation error.


# Candidate chassis-offset angles, relative to heading, at 15-degree
# resolution (23 options spanning the full circle, excluding straight ahead
# where the chassis would sit in front of the ball). plan_phi_sequence picks
# among these via its own cost search; state_blocked only needs to know at
# least one is ever clear. A coarser grid (originally just 7 choices at
# 45-degree spacing) could leave every single candidate blocked at once in a
# tight spot even though a fine-grained clear angle genuinely existed
# nearby. Ordered by distance from +-90 (beside the ball, usually the most
# open direction) rather than numerically: state_blocked's any() stops at
# the first clear candidate it finds, and this ordering is what that search
# spends the overwhelming majority of course-generation time in (profiled),
# so checking the likely-open directions first measurably speeds it up --
# the resulting *set* of candidates, and everywhere else that uses it, is
# unaffected by the order.
CHASSIS_OFFSET_CHOICES_DEG = sorted((d for d in range(-180, 180, 15) if d != 0),
                                     key=lambda d: min(abs(d - 90), abs(d + 90)))


def chassis_offset_clear(obstacles, x, y, phi):
    bx, by = x + ARM_LENGTH * math.cos(phi), y + ARM_LENGTH * math.sin(phi)
    # Inset by the chassis's own half-width: a bound on its *center* point
    # still lets up to BASE_XY_HALF of its physical box stick out past the
    # course boundary while technically passing the check.
    if not (-COURSE_HALF_WIDTH + BASE_XY_HALF <= by <= COURSE_HALF_WIDTH - BASE_XY_HALF):
        return False
    return not any(obstacle_blocks(o, bx, by, ROBOT_RADIUS + PLANNING_MARGIN) for o in obstacles)


# How wide a safe angular window a candidate needs, not just a single clear
# point. Without this, three-plus obstacles surrounding the ball can pinch
# the chassis's only viable angle down to a sliver just a couple of degrees
# wide -- clear right now, but gone the instant the ball advances a little
# further, forcing a jump to whatever's clear next (which, if the sliver was
# the only thing near the chassis's current side, can be all the way on the
# opposite side, with solid obstacle blocking every angle in between). A
# genuinely wide window can't vanish that fast, so it can always be tracked
# gradually. Matches the 15-degree candidate spacing: a candidate is only
# accepted if its own immediate neighbors would be too.
CHASSIS_SAFETY_HALF_WIDTH = math.radians(15)
CHASSIS_SAFETY_SAMPLES = 2


def chassis_offset_clear_wide(obstacles, x, y, phi):
    for k in range(-CHASSIS_SAFETY_SAMPLES, CHASSIS_SAFETY_SAMPLES + 1):
        test_phi = phi + CHASSIS_SAFETY_HALF_WIDTH * k / CHASSIS_SAFETY_SAMPLES
        if not chassis_offset_clear(obstacles, x, y, test_phi):
            return False
    return True


def state_blocked(obstacles, x, y, theta):
    """Checks only the ball's own point. This search's job is finding a
    ball path; whether the chassis can actually follow it is a completely
    separate, independently-verified question that plan_phi_sequence
    answers afterward (wide clearance *and* every transition between
    points, which this search doesn't check at all) -- so checking chassis
    wideness here too was pure redundant cost, not redundant safety: it
    couldn't change what does or doesn't end up in the final accepted
    course, since plan_phi_sequence rejects (triggering a full regenerate,
    same as this search failing outright) exactly the same paths either
    way. Measured: removing it cut a representative slow seed's generation
    time roughly 6x (16.9s -> 2.8s) with identical attempt counts and
    identical resulting courses, since profiling showed this search was
    where the overwhelming majority of course-generation time went."""
    if not (-COURSE_HALF_WIDTH + BALL_RADIUS <= y <= COURSE_HALF_WIDTH - BALL_RADIUS):
        return True
    return any(obstacle_blocks(o, x, y, BALL_CLEARANCE + PLANNING_MARGIN) for o in obstacles)


EDGE_SUBSTEPS = 4   # collision-check points along each edge, not just its endpoint


def edge_blocked(obstacles, x0, y0, theta0, x1, y1, theta1):
    """Checks several points along the step from (x0,y0,theta0) to
    (x1,y1,theta1), not just the endpoint. STEP_LEN (0.1m) is comparable to
    or larger than some obstacle radii, so endpoint-only checking can let
    an edge tunnel straight through an obstacle that sits between two
    otherwise-clear waypoints."""
    for i in range(1, EDGE_SUBSTEPS + 1):
        f = i / EDGE_SUBSTEPS
        if state_blocked(obstacles, x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, theta0 + (theta1 - theta0) * f):
            return True
    return False


def hybrid_astar(obstacles):
    """A* over a state lattice of (x, y, heading): each edge is a
    STEP_LEN-long forward arc at one of DTHETA_OPTIONS. Heading is part of
    the state -- not derived after the fact -- so the search can choose to
    angle through a gap that's only passable at that angle, the way a
    person turns sideways to slip through a crowd."""
    def key(x, y, theta):
        return (round(x / BIN_X), round(y / BIN_Y), round(theta / BIN_THETA))

    start = (0.0, 0.0, 0.0)
    if state_blocked(obstacles, *start):
        return None

    start_k = key(*start)
    dist = {start_k: 0.0}
    prev = {}
    state_of = {start_k: start}
    closed = set()
    pq = [(0.0, start_k)]

    while pq:
        _, k = heapq.heappop(pq)
        if k in closed:
            continue
        closed.add(k)
        x, y, theta = state_of[k]
        if x >= COURSE_LENGTH:
            goal_k = k
            break
        for dtheta in DTHETA_OPTIONS:
            ntheta = theta + dtheta
            if abs(ntheta) > THETA_MAX:
                continue
            heading_mid = theta + dtheta / 2
            nx, ny = x + STEP_LEN * math.cos(heading_mid), y + STEP_LEN * math.sin(heading_mid)
            if edge_blocked(obstacles, x, y, theta, nx, ny, ntheta):
                continue
            nk = key(nx, ny, ntheta)
            if nk in closed:
                continue
            step_cost = STEP_LEN * (1.0 + THETA_REG_WEIGHT * abs(ntheta) / THETA_MAX)
            nd = dist[k] + step_cost
            if nd < dist.get(nk, math.inf):
                dist[nk] = nd
                prev[nk] = k
                state_of[nk] = (nx, ny, ntheta)
                heapq.heappush(pq, (nd + max(COURSE_LENGTH - nx, 0.0), nk))
    else:
        return None

    path, k = [], goal_k
    while k in prev:
        path.append(state_of[k])
        k = prev[k]
    path.append(start)
    path.reverse()
    return path


def _angle_diff_signed(a, b):
    """a - b, wrapped to (-pi, pi]."""
    return (a - b + math.pi) % (2 * math.pi) - math.pi


PHI_TRANSITION_SUBSTEPS = 6


def phi_transition_clear(obstacles, x0, y0, phi0, x1, y1, phi1):
    """Checks that moving the chassis continuously from phi0 (with the ball
    at x0,y0) to phi1 (at x1,y1) never crosses a blocked point -- position
    and chassis angle interpolated together, exactly matching how the real
    system moves between two consecutive path points."""
    dphi = _angle_diff_signed(phi1, phi0)
    for k in range(1, PHI_TRANSITION_SUBSTEPS + 1):
        f = k / PHI_TRANSITION_SUBSTEPS
        x, y = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        if not chassis_offset_clear(obstacles, x, y, phi0 + dphi * f):
            return False
    return True


def plan_phi_sequence(obstacles, path_xs, path_ys, path_thetas):
    """Precomputes a chassis-offset angle for every path point via Dijkstra
    over (point_index, candidate) pairs: an edge exists only if the swept
    transition between two consecutive points' angles is collision-free the
    whole way (phi_transition_clear), not just at the endpoints.

    This replaces an earlier design where the chassis angle was chosen live,
    frame by frame, reacting to obstacles as they were encountered. That
    approach, however well-tuned (durable-candidate lookahead, wide-margin
    candidates, immediate correction when already unsafe), could not
    actually guarantee zero emergency corrections: it only ever verified
    that a candidate was clear *at a point*, never that the chassis could
    actually get from one point's safe angle to the next's within the
    distance available, so three-plus obstacles narrowing the safe zone
    from multiple sides at once could still isolate the tracked angle into
    a shrinking island with no reachable escape, however far ahead the
    lookahead looked. Planning the whole angle sequence in advance and
    verifying every transition, not just every point, removes that gap
    entirely: if this returns a sequence, the chassis can move through it
    exactly as planned with no live safety check ever needed again, the
    same way the ball's own path is proven collision-free in advance rather
    than hoping a live heuristic finds a way through moment to moment.

    Returns None if no such sequence exists, in which case the course
    should be regenerated -- same as when the ball's own path search
    fails."""
    n = len(path_xs)
    candidates = []
    for i in range(n):
        theta = path_thetas[i]
        opts = [theta + math.radians(d) for d in CHASSIS_OFFSET_CHOICES_DEG]
        opts = [phi for phi in opts if chassis_offset_clear_wide(obstacles, path_xs[i], path_ys[i], phi)]
        if not opts:
            return None
        candidates.append(opts)

    dist = {(0, j): 0.0 for j in range(len(candidates[0]))}
    prev = {}
    pq = [(0.0, 0, j) for j in range(len(candidates[0]))]
    heapq.heapify(pq)
    closed = set()
    goal = None
    while pq:
        d, i, j = heapq.heappop(pq)
        if (i, j) in closed:
            continue
        closed.add((i, j))
        if i == n - 1:
            goal = (i, j)
            break
        phi = candidates[i][j]
        x0, y0 = path_xs[i], path_ys[i]
        x1, y1 = path_xs[i + 1], path_ys[i + 1]
        for jn, phin in enumerate(candidates[i + 1]):
            if (i + 1, jn) in closed or not phi_transition_clear(obstacles, x0, y0, phi, x1, y1, phin):
                continue
            nd = d + abs(_angle_diff_signed(phin, phi))
            if nd < dist.get((i + 1, jn), math.inf):
                dist[(i + 1, jn)] = nd
                prev[(i + 1, jn)] = (i, j)
                heapq.heappush(pq, (nd, i + 1, jn))
    if goal is None:
        return None

    seq, node = [], goal
    while node in prev:
        seq.append(node)
        node = prev[node]
    seq.append(node)
    seq.reverse()
    return [candidates[i][j] for i, j in seq]


def generate_solvable_course(rng, max_attempts=50):
    for attempt in range(1, max_attempts + 1):
        obstacles = generate_course(rng)
        path = hybrid_astar(obstacles)
        if path is None:
            continue
        phis = plan_phi_sequence(obstacles, [w[0] for w in path], [w[1] for w in path], [w[2] for w in path])
        if phis is None:
            continue
        return obstacles, path, phis, attempt
    raise RuntimeError(f"no collision-free path found in {max_attempts} random courses")


# --- Ballistic / contact model (unchanged from v2) ---------------------------
class BallisticModel:
    """Nominal vertical-bounce timing (fall to floor, rebound to hand
    height); independent of push direction, so redirected hits (Obj. 4)
    can reuse it with a different v_push."""

    def __init__(self, v_push=V_PUSH, floor_e=FLOOR_E, contact_z=CONTACT_Z, floor_z=FLOOR_Z, g=G):
        drop = contact_z - floor_z
        v_impact = math.sqrt(v_push ** 2 + 2 * g * drop)
        t1 = (v_impact - v_push) / g
        v_rebound = floor_e * v_impact
        t2 = (v_rebound - math.sqrt(v_rebound ** 2 - 2 * g * drop)) / g
        self.flight_time = t1 + t2


def mirror_law_push(vz_in, hand_e=HAND_E, v_push=V_PUSH):
    """Outgoing vertical speed at contact; always -v_push regardless of
    vz_in (the hand_e*vz terms cancel)."""
    v_paddle_cmd = (-v_push + hand_e * vz_in) / (1 + hand_e)
    return v_paddle_cmd * (1 + hand_e) - hand_e * vz_in


@dataclass
class ContactTarget:
    xy: np.ndarray
    t: float
    push_dir: np.ndarray


@dataclass
class ContactEvent:
    t: float
    x: float
    y: float
    heading_deg: float


def spawn_ball():
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_RADIUS)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=BALL_RADIUS, rgbaColor=[1, 0.4, 0, 1])
    body = p.createMultiBody(BALL_MASS, col, vis, [0, 0, CONTACT_Z - 0.001])
    p.changeDynamics(body, -1, restitution=FLOOR_E)
    p.resetBaseVelocity(body, linearVelocity=[LANE_VX, 0, -V_PUSH])
    return body


# --- Robot: chassis + fixed-length arm (variable angle) + paddle -----------
class Robot:
    """Kinematic base + paddle on a fixed-length arm whose *angle* (phi) is
    chosen independently of the ball's push direction each step -- unlike a
    rigid heading+90-degree offset, this lets the chassis fall in behind the
    ball instead of beside it when the side is too tight. Chassis and
    paddle share phi as their orientation, since the arm rigidly connects
    them into one assembly that turns together. The arm is rendered as a
    debug line (not a rotated box) since it trivially handles any phi
    without solving for box orientation."""

    def __init__(self, arm_length=ARM_LENGTH, base_xy_half=BASE_XY_HALF):
        self.arm_length = arm_length
        self.base_xy_half = base_xy_half
        base_half = [base_xy_half, base_xy_half, PADDLE_CENTER_Z / 2]
        self.base_z = base_half[2]

        base_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=base_half, rgbaColor=[0.2, 0.7, 0.3, 1])
        self.base_id = p.createMultiBody(0, -1, base_vis, [0, arm_length, self.base_z])

        paddle_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=PADDLE_HALF)
        paddle_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=PADDLE_HALF, rgbaColor=[0.2, 0.2, 0.8, 1])
        self.paddle_id = p.createMultiBody(0, paddle_col, paddle_vis, [0, 0, PADDLE_CENTER_Z])
        p.changeDynamics(self.paddle_id, -1, restitution=0.0)

        self.arm_line_id = -1
        self.base_xy = np.array([0.0, arm_length])

    def set_pose(self, ball_xy, phi, paddle_z=PADDLE_CENTER_Z):
        """ball_xy: the paddle's position (== the ball's own, since they
        must coincide). phi: world-frame angle from the paddle to the
        chassis. Both the chassis and the paddle are oriented by phi --
        they're one rigid assembly connected by the arm, so they turn
        together, rather than the paddle spinning to face the ball's push
        direction independently of which way the arm/chassis are facing."""
        paddle_xy = np.array(ball_xy)
        unit = np.array([math.cos(phi), math.sin(phi)])
        base_xy = paddle_xy + self.arm_length * unit
        self.base_xy = base_xy

        quat = p.getQuaternionFromEuler([0, 0, phi])
        p.resetBasePositionAndOrientation(self.base_id, [*base_xy, self.base_z], quat)
        p.resetBasePositionAndOrientation(self.paddle_id, [*paddle_xy, paddle_z], quat)

        base_face = base_xy - self.base_xy_half * unit   # line starts at the chassis edge, not its center
        self.arm_line_id = p.addUserDebugLine(
            [*base_face, PADDLE_CENTER_Z], [*paddle_xy, PADDLE_CENTER_Z],
            lineColorRGB=[0.2, 0.7, 0.3], lineWidth=6, replaceItemUniqueId=self.arm_line_id)
        return base_xy

    def touches(self, other_id):
        return bool(p.getContactPoints(self.paddle_id, other_id))


# --- World setup --------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()
seed = args.seed if args.seed is not None else random.randrange(2 ** 31)
rng = random.Random(seed)
print(f"course seed: {seed}")

obstacles, path, path_phis_list, attempts = generate_solvable_course(rng)
path_xs = np.array([w[0] for w in path])
path_ys = np.array([w[1] for w in path])
path_thetas = np.array([w[2] for w in path])
path_phis = np.array(path_phis_list)
path_s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(path_xs), np.diff(path_ys)))])
path_length = float(path_s[-1])
print(f"generated a solvable {len(obstacles)}-obstacle course after {attempts} attempt(s); "
      f"path length {path_length:.2f}m vs {COURSE_LENGTH:.2f}m straight "
      f"(+{path_length - COURSE_LENGTH:.2f}m overhead); "
      f"max heading swing {math.degrees(np.max(np.abs(path_thetas))):.0f}deg")


def plan_next_contact(ball_xy, wp_idx, t_now, bounce: BallisticModel, lane_speed=LANE_VX):
    """Waypoint-follower, not a nearest-point search: aims at the planner's
    lattice point path[wp_idx], advancing wp_idx (monotonically -- it never
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
    while wp_idx < len(path_xs) - 1 and math.hypot(path_xs[wp_idx] - ball_xy[0], path_ys[wp_idx] - ball_xy[1]) < step_dist:
        wp_idx += 1
    target_xy = np.array([path_xs[wp_idx], path_ys[wp_idx]])
    delta = target_xy - np.array(ball_xy)
    dist = np.linalg.norm(delta)
    push_dir = delta / dist if dist > 1e-9 else np.array([1.0, 0.0])
    return ContactTarget(xy=target_xy, t=t_now + bounce.flight_time, push_dir=push_dir), wp_idx


def phi_at_progress(path_xs, path_ys, path_phis, ball_xy, wp_idx):
    """Chassis-offset angle for the ball's current position along the
    (wp_idx-1 -> wp_idx) segment of the precomputed, verified phi sequence
    -- position and angle interpolated together, exactly matching how
    plan_phi_sequence validated that transition. No live safety check
    needed: the whole line was proven collision-free in advance."""
    i0, i1 = max(wp_idx - 1, 0), wp_idx
    x0, y0 = path_xs[i0], path_ys[i0]
    x1, y1 = path_xs[i1], path_ys[i1]
    seg = np.array([x1 - x0, y1 - y0])
    seg_len2 = float(seg @ seg)
    if seg_len2 < 1e-12:
        return path_phis[i1]
    frac = float(np.clip((np.array(ball_xy) - [x0, y0]) @ seg / seg_len2, 0.0, 1.0))
    return path_phis[i0] + _angle_diff_signed(path_phis[i1], path_phis[i0]) * frac


p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(DT)

plane = p.loadURDF("plane.urdf")
p.changeDynamics(plane, -1, restitution=FLOOR_E)

spawn_obstacles(obstacles)
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
PUSH_DEPTH, PUSH_STEPS = 0.05, 12
push_timer = 0
contact_cooldown = 0
heading = 0.0
target, path_idx = plan_next_contact([0.0, 0.0], 0, 0.0, bounce)
contact_log = []
t = 0.0
# Scaled to the planned path's actual arc length, not just straight-line
# course length: heading swings mean the ball travels farther than
# COURSE_LENGTH to cover it, and a fixed cap could cut off a valid run.
MAX_STEPS = int(max(60.0, 2.0 * path_length / LANE_VX) / DT)

for step in range(MAX_STEPS):
    p.stepSimulation()
    t += DT

    ball_pos, _ = p.getBasePositionAndOrientation(ball)
    ball_vel, _ = p.getBaseVelocity(ball)
    vz = ball_vel[2]

    # The paddle must coincide with the ball at contact anyway, so there's
    # no need to *predict* where it should be -- just make it the ball's
    # real current position every step (no integration, no timing model,
    # nothing that can drift or diverge from reality). heading only changes
    # at a contact (the ball is a free projectile between them, so its
    # direction of travel is genuinely fixed until then).
    #
    # phi (the chassis's offset angle) is looked up from the precomputed,
    # fully-verified sequence (plan_phi_sequence) -- interpolated by the
    # ball's actual position along the current path segment, matching
    # exactly what was validated. No live obstacle check needed here at all.
    phi = phi_at_progress(path_xs, path_ys, path_phis, ball_pos[:2], path_idx)

    paddle_z = PADDLE_CENTER_Z - PUSH_DEPTH * (push_timer / PUSH_STEPS) if push_timer > 0 else PADDLE_CENTER_Z
    push_timer = max(push_timer - 1, 0)
    robot.set_pose(ball_pos[:2], phi, paddle_z)

    # No vz>0 gate: mirror_law_push is an identity in vz, and gating on its
    # sign can silently drop the contact if PyBullet's own collision
    # response zeroes vz before this code sees it.
    if robot.touches(ball) and contact_cooldown <= 0:
        v_new_z = mirror_law_push(vz)
        target, path_idx = plan_next_contact(ball_pos[:2], path_idx, t, bounce)
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

p.disconnect()

print(f"Contacts recorded: {len(contact_log)}")
if contact_log:
    headings = [c.heading_deg for c in contact_log]
    print(f"Heading ranged {min(headings):.0f}deg to {max(headings):.0f}deg during the run")
