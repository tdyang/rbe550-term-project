"""Random obstacle-course generation, spawning, and the point-vs-obstacle
collision primitive the planner is built on."""

import math
from dataclasses import dataclass

import pybullet as p

from config import ARM_LENGTH, COURSE_HALF_WIDTH, COURSE_LENGTH, N_OBSTACLES_RANGE


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


# Random size ranges per obstacle kind, in meters -- (hx, hy, height), where
# hx/hy are half-extents for crate/wall but hx doubles as the radius (hy
# unused) for pillar. Centralized here, rather than inlined in
# generate_course, so other tools (e.g. a reference figure of what each kind
# looks like and how big it can get) can report these exact ranges instead
# of a second, driftable copy. Sized at or above the robot's own footprint
# (BASE_XY_HALF=0.2) so obstacles are genuine obstacles, not gaps the robot
# dwarfs; wall is thin but at least robot-length along its long axis.
OBSTACLE_SIZE_RANGES = {
    "crate":  {"hx": (0.20, 0.30), "hy": (0.20, 0.30), "height": (0.2, 0.4)},
    "pillar": {"hx": (0.25, 0.40), "hy": (0.25, 0.40), "height": (0.4, 0.8)},   # wider cylinders
    "wall":   {"hx": (0.06, 0.10), "hy": (0.25, 0.45), "height": (0.3, 0.6)},
}


def generate_course(rng):
    """6-9 obstacles, each with a random kind/size/orientation, scattered on
    both sides of the straight-ahead line (y drawn from a triangular
    distribution peaked at 0) so the path usually has to weave rather than
    just lean to one side."""
    n = rng.randint(*N_OBSTACLES_RANGE)
    margin = 0.6
    # Wide enough that any gap between two obstacles has room for the robot
    # body *and* the chassis's ARM_LENGTH swing, not just the chassis alone
    # -- at the old 0.15m, two obstacles could still randomly land ~0.5m
    # apart (ball clearance alone is ~0.34m of that), leaving no room for
    # the chassis to take either side, which forced last-instant, visibly
    # jarring corrections.
    min_center_gap = 2.0 * ARM_LENGTH
    obstacles = []
    for _ in range(n):
        yaw = rng.uniform(0, 2 * math.pi)
        kind = rng.choice(list(OBSTACLE_SIZE_RANGES))
        size = OBSTACLE_SIZE_RANGES[kind]
        if kind == "pillar":
            hx = hy = rng.uniform(*size["hx"])
        else:
            hx, hy = rng.uniform(*size["hx"]), rng.uniform(*size["hy"])
        height = rng.uniform(*size["height"])

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
