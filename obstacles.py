"""Random obstacle-course generation, spawning, and the point-vs-obstacle
collision primitive the planner is built on."""

import math
from dataclasses import dataclass

import numpy as np

import pybullet as p

from config import (
    ARM_LENGTH, BASE_XY_HALF, COURSE_HALF_WIDTH, COURSE_LENGTH, DYNAMIC_AMPLITUDE_RANGE,
    DYNAMIC_OBSTACLE_FRACTION, DYNAMIC_PERIOD_RANGE, N_OBSTACLES_RANGE,
    ROAM_SPEED_JITTER, ROAM_SPEED_RANGE, ROAM_TURN_MAX, ROAM_TURN_RATE, ROAM_X_RANGE,
)


@dataclass
class Obstacle:
    kind: str    # 'crate' | 'pillar' | 'wall'
    x: float     # rest position (center of its oscillation, if moving)
    y: float
    yaw: float
    hx: float    # half-extent (box) or radius (pillar)
    hy: float    # half-extent (box); unused for pillar
    height: float
    moving: bool = False
    amplitude: float = 0.0   # meters, peak offset from (x, y) along world y
    period: float = 1.0      # seconds, full back-and-forth cycle
    phase: float = 0.0       # radians, so obstacles don't all move in lockstep
    # Roaming motion (motion="roam" courses): a precomputed (N, 2) table of
    # center positions every ROAM_DT seconds (simulate_roaming), replacing
    # the sinusoid. None for oscillating/static obstacles.
    traj: np.ndarray = None
    speed: float = 0.0       # roaming speed, m/s (constant; bounces are elastic)

    def __post_init__(self):
        # obstacle_blocks runs tens of millions of times during planning;
        # precomputing its per-obstacle constants once here (rather than
        # recomputing sin/cos and a reach bound on every single call) is a
        # meaningful, purely mechanical speedup with no effect on what it
        # computes.
        self.cos_nyaw = math.cos(-self.yaw)
        self.sin_nyaw = math.sin(-self.yaw)
        self.reach = self.hx + self.hy   # conservative (>= true half-diagonal) bound for a cheap early reject
        # Upper bound on how fast this obstacle's center ever moves (the
        # sinusoid's peak slope), so a planner can reserve room for timing
        # error: within +-dt of any instant it can't have moved farther
        # than peak_speed*dt.
        self.peak_speed = 2 * math.pi * self.amplitude / self.period if self.moving else 0.0

    @property
    def radius(self):
        """Bounding-circle radius, used for roaming obstacles' bounces."""
        return math.hypot(self.hx, self.hy)

    def position_at(self, t):
        """Where this obstacle's center actually is at time t. Static
        obstacles (moving=False) ignore t entirely, so every existing
        caller that never passes a time is unaffected. Motion is a plain
        sinusoid along world y only (not yaw, not x) -- simple enough that
        a planner can reason about "where will this be at time t" in
        closed form, which is the whole point: it lets moving obstacles be
        proven collision-free in advance exactly like static ones, rather
        than requiring a live reactive layer."""
        if not self.moving:
            return self.x, self.y
        if self.traj is not None:
            f = min(max(t / ROAM_DT, 0.0), len(self.traj) - 1.000001)
            i = int(f)
            f -= i
            (x0, y0), (x1, y1) = self.traj[i], self.traj[i + 1]
            return x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        return self.x, self.y + self.amplitude * math.sin(2 * math.pi * t / self.period + self.phase)


def obstacle_blocks(obs, x, y, radius, t=0.0, time_margin=0.0):
    """True if a disc of `radius` at (x, y) overlaps `obs` at any time
    within t +- time_margin (both ignored for a static obstacle), by
    inflating `radius` by the farthest the obstacle can move in that
    window (obs.peak_speed * time_margin). See shape_blocks for the
    geometry."""
    cx, cy = obs.position_at(t)
    return shape_blocks(obs, cx, cy, x, y, radius + obs.peak_speed * time_margin)


def shape_blocks(obs, cx, cy, x, y, radius):
    """True if a disc of `radius` at (x, y) overlaps `obs`'s shape centered
    at (cx, cy) -- obstacle_blocks passes where the obstacle really is;
    the live planner passes where its sensing *predicts* it will be.
    Pillars: circle test. Crates/walls: distance to an axis-aligned
    rectangle in the obstacle's own rotated frame (Minkowski sum of a
    rotated rect and a disc is a rounded rect). Called tens of millions of
    times during planning, hence: squared-distance comparisons throughout
    (no sqrt), a cheap early reject using obs.reach, and
    obs.cos_nyaw/sin_nyaw computed once per obstacle
    (Obstacle.__post_init__) rather than every call."""
    dx, dy = x - cx, y - cy
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


ROAM_DT = 0.05        # seconds between samples of a roaming obstacle's trajectory table
ROAM_HORIZON = 300.0  # seconds of trajectory precomputed; positions clamp after that


def roaming_step(radii, pos, vel, statics, dt, pushers=(), shapes=None):
    """One step of roaming motion for movers with bounding radii `radii`,
    positions `pos`, and velocities `vel` (parallel lists of 2-vectors):
    straight lines, bouncing elastically off the course edges
    (y = +-COURSE_HALF_WIDTH, x = ROAM_X_RANGE), the `statics`, and each
    other (equal masses: swap the normal velocity components), using
    bounding circles for contact. Returns the new positions and updates
    `vel` in place. simulate_roaming drives the real obstacles with it,
    and the live planner's predictor rolls the same rule forward from what
    it senses (live_planning.ObstacleSensor).

    `pushers`: (center, radius) circles that move on their own -- the
    robot's chassis, during a run (RoamingField). An obstacle reaching one
    turns away at its own speed, like a pedestrian bumping off someone,
    and is set just clear of it: holding it in place, as with a static
    obstacle, would let a pusher that keeps coming sink into it. Contact
    with a pusher uses the mover's true shape (`shapes`, the Obstacles):
    a long wall's bounding circle reaches well past the wall itself, and
    bounced obstacles off a robot the wall never touched."""
    cand = []
    for i, r in enumerate(radii):
        p, v = pos[i] + vel[i] * dt, vel[i].copy()
        for axis, lo, hi in ((0, ROAM_X_RANGE[0] + r, ROAM_X_RANGE[1] - r),
                             (1, -COURSE_HALF_WIDTH + r, COURSE_HALF_WIDTH - r)):
            if p[axis] < lo or p[axis] > hi:
                p[axis] = min(max(p[axis], lo), hi)
                v[axis] = abs(v[axis]) if p[axis] == lo else -abs(v[axis])
        for s in statics:
            if np.linalg.norm(p - (s.x, s.y)) < r + s.radius:
                # Reflect and hold this step rather than pushing it back
                # out -- a push-out teleports an obstacle that's wedged
                # between two things. The normal is taken at the old
                # (still clear) position: a step that newly overlaps is
                # always heading in along that one, whereas at the new
                # position a glancing step can look like it's already
                # heading out and slip a few millimeters inside.
                d = pos[i] - (s.x, s.y)
                n = d / np.linalg.norm(d)
                if v @ n < 0:
                    v = v - 2 * (v @ n) * n
                p = pos[i]
        for c, rc in pushers:
            if not shape_blocks(shapes[i], p[0], p[1], c[0], c[1], rc):
                continue
            d = p - c
            dist = float(np.linalg.norm(d))
            n = d / dist if dist > 1e-9 else np.array([1.0, 0.0])
            if v @ n < 0:
                v = v - 2 * (v @ n) * n
            for _ in range(40):   # step out along the normal until just clear
                p = p + 0.005 * n
                if not shape_blocks(shapes[i], p[0], p[1], c[0], c[1], rc):
                    break
        cand.append(p)
        vel[i] = v
    # Movers against each other, checked on where they're *about* to be:
    # any pair that would touch bounces and both hold this step, so they
    # never pass into each other. A hold can put a pair back in the way of
    # a third mover's candidate, so repeat until nothing changes; holds
    # only ever remove motion, so this settles.
    changed = True
    while changed:
        changed = False
        for i in range(len(radii)):
            for j in range(i + 1, len(radii)):
                if np.linalg.norm(cand[j] - cand[i]) >= radii[i] + radii[j]:
                    continue
                d = pos[j] - pos[i]   # old positions, as with statics above
                n = d / np.linalg.norm(d)
                rel = (vel[i] - vel[j]) @ n
                if rel > 0:   # approaching
                    vel[i] = vel[i] - rel * n
                    vel[j] = vel[j] + rel * n
                for k in (i, j):
                    if cand[k] is not pos[k]:
                        cand[k] = pos[k]
                        changed = True
    return cand


ROBOT_CONTACT_RADIUS = BASE_XY_HALF * math.sqrt(2)   # the chassis's bounding circle, for obstacles bouncing off it


class RoamingField:
    """Steps the roaming obstacles live during a run, ROAM_DT at a time,
    so they bounce off the robot's chassis (roaming_step's pushers) rather
    than passing through it. It restarts each obstacle's trajectory table
    (obs.traj, which position_at reads) from its initial state and extends
    it as the run goes, keeping one tick ahead of the clock for
    interpolation. With nothing touching the robot this reproduces the
    precomputed table (simulate_roaming) exactly, so the offline planner's
    perfect-knowledge plans -- which keep clear of every obstacle -- stay
    valid."""

    def __init__(self, obstacles, robot_xy=(0.0, 0.0)):
        self.movers = [o for o in obstacles if o.traj is not None]
        self.statics = [o for o in obstacles if not o.moving]
        self.radii = [o.radius for o in self.movers]
        self.pos = [np.array(o.traj[0], dtype=float) for o in self.movers]
        self.vel = [o.vel0.copy() for o in self.movers]
        for o, p in zip(self.movers, self.pos):
            o.traj = [p.copy()]
        self.t = 0.0
        self.advance(0.0, robot_xy)   # one tick ahead from the start, so position_at(0) can interpolate

    def advance(self, t, robot_xy):
        while self.t < t + ROAM_DT:
            apply_turns(self.movers, self.vel, len(self.movers[0].traj) - 1 if self.movers else 0)
            pushers = [(np.asarray(robot_xy, dtype=float), ROBOT_CONTACT_RADIUS)]
            self.pos = roaming_step(self.radii, self.pos, self.vel, self.statics, ROAM_DT, pushers, self.movers)
            for o, p, v in zip(self.movers, self.pos, self.vel):
                o.traj.append(p.copy())
                o.peak_speed = max(o.peak_speed, float(np.linalg.norm(v)))
            self.t += ROAM_DT


def apply_turns(movers, vel, tick):
    """Applies any random turn scheduled for this tick (simulate_roaming's
    obs.turns) to the movers' velocities, in place."""
    for i, o in enumerate(movers):
        turn = o.turns.get(tick)
        if turn is not None:
            dtheta, speed = turn
            c, s = math.cos(dtheta), math.sin(dtheta)
            v = vel[i]
            heading = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
            vel[i] = speed * heading / max(float(np.linalg.norm(heading)), 1e-9)


def simulate_roaming(obstacles, rng, horizon=ROAM_HORIZON, dt=ROAM_DT):
    """Gives every moving obstacle a random heading, its own base speed
    from ROAM_SPEED_RANGE, and a schedule of random turns (ROAM_TURN_RATE,
    varying speed around that base by ROAM_SPEED_JITTER), then steps all
    of them forward together (apply_turns, roaming_step). Stores each
    one's positions in obs.traj, so position_at(t) is a lookup -- the
    same table drives the live PyBullet bodies and the offline planner's
    perfect-knowledge baseline, while the live planner only ever senses
    the current instant of it (live_planning.ObstacleSensor)."""
    movers = [o for o in obstacles if o.moving]
    statics = [o for o in obstacles if not o.moving]
    pos = {id(o): np.array([o.x, o.y]) for o in movers}
    vel = {}
    for o in movers:
        o.speed = rng.uniform(*ROAM_SPEED_RANGE)
        heading = rng.uniform(0, 2 * math.pi)
        vel[id(o)] = o.speed * np.array([math.cos(heading), math.sin(heading)])
    trajs = {id(o): [pos[id(o)].copy()] for o in movers}
    for o in movers:
        o.vel0 = vel[id(o)].copy()   # where RoamingField starts a live run from
        # Random turns, drawn once here so the precomputed table and a live
        # run (RoamingField) apply exactly the same ones: tick -> (heading
        # change, new speed).
        o.turns, t_turn = {}, rng.expovariate(ROAM_TURN_RATE) if ROAM_TURN_RATE > 0 else math.inf
        while t_turn < horizon:
            o.turns[int(t_turn / dt)] = (rng.uniform(-ROAM_TURN_MAX, ROAM_TURN_MAX),
                                         o.speed * rng.uniform(1 - ROAM_SPEED_JITTER, 1 + ROAM_SPEED_JITTER))
            t_turn += rng.expovariate(ROAM_TURN_RATE)

    peak = {id(o): o.speed for o in movers}
    radii = [o.radius for o in movers]
    p, v = [pos[id(o)] for o in movers], [vel[id(o)] for o in movers]
    for tick in range(int(horizon / dt)):
        apply_turns(movers, v, tick)
        p = roaming_step(radii, p, v, statics, dt)
        for o, pi, vi in zip(movers, p, v):
            trajs[id(o)].append(pi.copy())
            peak[id(o)] = max(peak[id(o)], float(np.linalg.norm(vi)))
    for o in movers:
        o.traj = np.array(trajs[id(o)])
        # Equal-mass bounces exchange speed between obstacles, so the peak
        # over the whole run (not the starting speed) bounds how far one
        # can move in a given time.
        o.peak_speed = peak[id(o)]


# Minimum clearance between any two obstacles' bounding circles (on top of
# both radii, and any oscillation amplitude). It used to be 2*ARM_LENGTH
# (1m) so that every gap fit the robot and its arm -- but at that spacing
# 6-9 obstacles almost never fit in the lane, and the generator was
# silently placing the leftovers inside other obstacles (2-5 actually fit
# once that was fixed). Gaps narrower than the robot don't make a course
# unsolvable; the planner routes another way or the course is regenerated.
# Measured over 200 courses: 0.4m gives ~6.4 obstacles per roaming course
# (4-8) and ~4.3 per oscillating one (each oscillating obstacle also
# reserves its whole sweep), versus 3.5 and 2.8 at 1m.
MIN_OBSTACLE_GAP = 0.4


def generate_course(rng, motion="oscillate"):
    """6-9 obstacles, each with a random kind/size/orientation, scattered on
    both sides of the straight-ahead line (y drawn from a triangular
    distribution peaked at 0) so the path usually has to weave rather than
    just lean to one side. A DYNAMIC_OBSTACLE_FRACTION of them also get a
    fixed sinusoidal back-and-forth motion (Obstacle.position_at) rather
    than sitting still -- or, with motion="roam", roam the course instead
    (simulate_roaming), starting from inside ROAM_X_RANGE."""
    n = rng.randint(*N_OBSTACLES_RANGE)
    margin = 0.6
    min_center_gap = MIN_OBSTACLE_GAP
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

        moving = rng.random() < DYNAMIC_OBSTACLE_FRACTION
        oscillates = moving and motion == "oscillate"
        amplitude = rng.uniform(*DYNAMIC_AMPLITUDE_RANGE) if oscillates else 0.0
        period = rng.uniform(*DYNAMIC_PERIOD_RANGE) if oscillates else 1.0
        phase = rng.uniform(0, 2 * math.pi) if oscillates else 0.0

        # Both bounds and inter-obstacle spacing are padded by amplitude (on
        # top of the usual reach/min_center_gap), since a moving obstacle
        # must stay in-bounds and collision-free with its neighbors across
        # its whole swept range, not just at its rest position -- otherwise
        # a course could generate "solvable" only to have a moving obstacle
        # sweep out of bounds or through a neighbor mid-run.
        reach = math.hypot(hx, hy)
        swept_reach = reach + amplitude
        y_lo, y_hi = -COURSE_HALF_WIDTH + swept_reach, COURSE_HALF_WIDTH - swept_reach

        x_lo, x_hi = margin, COURSE_LENGTH - margin
        if moving and motion == "roam":
            x_lo, x_hi = ROAM_X_RANGE[0] + reach, ROAM_X_RANGE[1] - reach
        for _attempt in range(200):
            x = rng.uniform(x_lo, x_hi)
            y = rng.triangular(y_lo, y_hi, 0.0) if y_hi > y_lo else 0.0
            if all(math.hypot(x - o.x, y - o.y) >= swept_reach + math.hypot(o.hx, o.hy) + o.amplitude + min_center_gap
                   for o in obstacles):
                obstacles.append(Obstacle(kind, x, y, yaw, hx, hy, height, moving, amplitude, period, phase))
                break
        # No spot found in 200 tries: leave this one out rather than placing
        # it at the last rejected spot, which could put it inside another.
    if motion == "roam":
        simulate_roaming(obstacles, rng)
    return obstacles


def spawn_obstacles(obstacles, t=0.0):
    """Spawns each obstacle at its position at time t (t=0 by default,
    which is also every static obstacle's position at any time)."""
    color = {"crate": [0.6, 0.4, 0.2, 1], "pillar": [0.5, 0.5, 0.55, 1], "wall": [0.7, 0.2, 0.2, 1]}
    moving_tint = [1.0, 1.0, 0.3, 1]   # a faint yellow wash so moving obstacles read as distinct on sight
    body_ids = []
    for obs in obstacles:
        quat = p.getQuaternionFromEuler([0, 0, obs.yaw])
        z = obs.height / 2
        rgba = color[obs.kind]
        if obs.moving:
            rgba = [min(1.0, 0.6 * c + 0.4 * m) for c, m in zip(rgba, moving_tint)]
        if obs.kind == "pillar":
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=obs.hx, height=obs.height)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=obs.hx, length=obs.height, rgbaColor=rgba)
        else:
            half = [obs.hx, obs.hy, z]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba)
        x, y = obs.position_at(t)
        body_ids.append(p.createMultiBody(0, col, vis, [x, y, z], quat))
    return body_ids


def move_obstacles(obstacles, body_ids, t):
    """Repositions every moving obstacle's already-spawned body to its
    position at time t; static obstacles are skipped (their body never
    needs to move). Call once per simulation step with the real elapsed
    time so the live obstacles track exactly what the planner assumed."""
    for obs, bid in zip(obstacles, body_ids):
        if not obs.moving:
            continue
        x, y = obs.position_at(t)
        quat = p.getQuaternionFromEuler([0, 0, obs.yaw])
        p.resetBasePositionAndOrientation(bid, [x, y, obs.height / 2], quat)
