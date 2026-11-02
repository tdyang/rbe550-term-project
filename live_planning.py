"""Live (online) planner for roaming obstacles: replans at every dribble
contact from what it can sense *now*, rather than proving the whole run in
advance the way planning.py does.

Why a different planner: planning.py's guarantee rests on knowing every
obstacle's position at every future time (Obstacle.position_at), which is
only true for obstacles on a fixed, known schedule. Roaming obstacles
(obstacles.simulate_roaming) bounce off each other and the course, so the
planner here is only allowed to sense each obstacle's current position
(ObstacleSensor), estimate its velocity from successive readings, and
replan as those readings change -- the proposal's stretch goal 1, "online
re-planning rather than a single offline course map". The cost is the
guarantee: a run can no longer be proven collision-free before it starts,
so collisions become a measured outcome (simulation.py counts them).

Per contact (~4 Hz):
  1. Sense obstacles; predict each moving one forward at constant velocity.
     Grid cells the robot body can't occupy now or within HARD_HORIZON are
     blocked; cells the prediction sweeps through later, out to
     PREDICT_HORIZON, only get a penalty that fades with lead time -- a
     constant-velocity guess a second out is a reason to prefer another
     way, not a wall. (Hard-blocking the whole horizon was tried first: on
     these dense courses two or three predicted sweeps routinely closed
     off the course's entire width, and the robot stood at the start
     reporting no path while obstacles ran into it.)
  2. D* Lite (dstar_lite.py) repairs its body-center path to the finish
     line, touching only cells whose occupancy or penalty changed since
     the last contact. If there is no path and an obstacle is closing in
     on the robot, it steps away from it rather than standing still.
  3. The body steps along that path; the arm angle is chosen locally among
     the same candidates planning.py uses, keeping the ball clear of the
     predicted obstacles and within BALL_SPEED_MAX of where it is.
"""

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from config import (
    ARM_LENGTH, BALL_CLEARANCE, BALL_RADIUS, BALL_SPEED_MAX, BASE_SPEED_MAX, BASE_TURN_RATE_MAX,
    BASE_XY_HALF, COURSE_HALF_WIDTH, COURSE_LENGTH, ROBOT_RADIUS,
)
from dstar_lite import DStarLite
from obstacles import ROAM_DT, roaming_step, shape_blocks
from planning import (
    CHASSIS_OFFSET_CHOICES_DEG, LANDING_MARGIN, PLANNING_MARGIN, START_SLACK, THETA_MAX, angle_diff_signed,
    ball_in_bounds, clearance_at, cruise_speed,
    hybrid_astar, paddle_xy,
)

GRID_RES = 0.1                          # meters per D* Lite cell
GRID_X0, GRID_X1 = -0.6, COURSE_LENGTH + 0.3
GRID_Y0 = -COURSE_HALF_WIDTH
HARD_HORIZON = 0.4                      # seconds of predicted sweep that are hard-blocked
PREDICT_HORIZON = 2.5                   # seconds of predicted sweep that are penalized
PREDICT_DT = 0.2                        # spacing of predicted samples along that horizon
PREDICT_PENALTY = 4.0                   # extra cost multiplier for a cell swept right after HARD_HORIZON,
                                        # fading linearly to 0 at PREDICT_HORIZON
BALL_PREDICT_HORIZON = 1.0              # how long after landing the ball must stay clear of predicted obstacles
LOOKAHEAD = 0.3                         # meters along the D* Lite path the body steers toward
ARM_SIDE_WEIGHT = 0.3                   # preference for the ball beside the body (+-90 deg)
CURRENT_PENALTY = 8.0                   # "dwa": soft cost of a moving obstacle's near-term sweep

# Live hybrid A* ("hybrid"): node-expansion budget per replan, so a boxed-in
# search gives up in bounded time instead of exhausting the lattice.
HYBRID_MAX_EXPANSIONS = 20000

# Dynamic-window local step ("dwa"): candidate body velocities
# (dwa_velocities), each rolled out at constant velocity over DWA_HORIZON
# against the predicted obstacles.
EVADE_SPEED = min(0.9 * BALL_SPEED_MAX, BASE_SPEED_MAX)
DWA_DIRECTIONS = [np.array([math.cos(a), math.sin(a)]) for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
CLEARANCE_LOOKAHEAD = 0.6   # meters of route ahead whose free space sets the cruise speed

# Being chased: a moving obstacle behind the robot (relative to where it's
# heading) and closing on it fast enough to reach it within CHASE_HORIZON.
# Slowing down for a narrow way only helps the chaser, so the robot runs
# at EVADE_SPEED instead, with no extra cost for the speed. Measured over 90
# courses it only pays off with the faster open-ground cruise: at a 0.35
# m/s top speed it cost 5 finishes (78 vs. 83/90); at 0.45 it cut
# collisions by a quarter (72 vs. 97) and added 2 finishes.
CHASE_SPEEDUP = True
CHASE_HORIZON = 2.0         # seconds until a closing obstacle would reach the robot
CHASE_RANGE = 1.5           # meters; farther obstacles aren't treated as chasing yet

# Shielding the ball: like a player keeping their body between the ball and
# a defender, penalize arm angles that put the ball on the side of the
# nearest moving obstacle within SHIELD_RANGE (score units: radians of arm
# turn). 0 disables it. Not adopted: over 90 courses at weight 1.0 it gained
# 2 finishes at a 0.35 m/s top speed (85 vs. 83/90) but lost 4 at 0.45.
SHIELD_WEIGHT = 0.0
SHIELD_RANGE = 1.0          # meters of gap to the robot body

# Keeping the ball in bounds. The boundary is a hard limit on every landing
# spot (including the least-bad fallback when nothing is clear of
# obstacles), and the dynamic window only moves the body where some
# reachable arm angle keeps the ball inside. On top of that, landing spots
# within WALL_MARGIN of a side line are penalized, so the arm swings the
# ball inward *before* the turn-rate limit can trap it on the wall side --
# which is how it was being bounced out of bounds in 13 of 30 runs.
WALL_MARGIN = 0.25          # meters from a side line
WALL_WEIGHT = 0.5           # score units (radians of arm turn) at the line itself


def dwa_velocities(cruise):
    """Standing still, plus 16 directions at this contact's cruise speed
    and at the evasive speed."""
    return [np.zeros(2)] + [speed * d for speed in (cruise, EVADE_SPEED) for d in DWA_DIRECTIONS]
DWA_HORIZON = 2.0           # seconds of rollout
DWA_STATIC_HORIZON = 0.4    # a static obstacle within this much rollout rules a velocity out
# Tuned on 30 roaming courses (6.4 obstacles, 3.2 moving): rejecting
# collisions within 1.0s (was 0.6s) and valuing clearance up to 0.5m at
# double weight (was 0.3m at 0.5) took the robot from 25 to 28 finishes and
# 47 to 27 collisions, by making it give way earlier. Making backing up
# cheaper or looking 2s ahead did no better: stepping aside or waiting is
# usually the better escape.
#
# Look-ahead was later lengthened (rollout 1.2 -> 2.0s, route penalties
# 1.5 -> 2.5s, ball landing check 0.5 -> 1.0s) once prediction became
# bounce-aware (PREDICTION): with straight-line prediction a 2s rollout did
# worse, since it only extended a bad guess, but with accurate prediction
# it cut collisions from 50 to 37 over 90 courses (75 -> 76 finished).
DWA_MIN_TTC = 1.0           # a predicted collision sooner than this rules a velocity out
DWA_SAFE_GAP = 0.5          # clearance (m) beyond which more doesn't score higher
DWA_CLEARANCE_WEIGHT = 1.0
DWA_TTC_WEIGHT = 2.0
DWA_FAST_COST = 0.3         # evasive speed only when it's worth more than this


# How the planner predicts a moving obstacle's next few seconds from what it
# senses: "bounce" rolls the obstacles' motion rule (obstacles.roaming_step:
# straight lines, elastic bounces off the course edges, static obstacles,
# and each other) forward from the sensed positions and velocities;
# "constant" extrapolates each one in a straight line, blind to bounces.
# Either way it predicts from the present -- it never reads an obstacle's
# actual future -- but "bounce" does assume the robot knows how obstacles
# bounce.
# Over 90 courses, "bounce" finished 75 against 72 for "constant", halved
# the runs ending out of bounds (6 -> 3), and had collisions in 19 runs
# rather than 25; its predictions 2s ahead are off by 0.3cm on average,
# against 12cm (54cm at the 95th percentile) for "constant".
PREDICTION = "bounce"
PREDICT_TRACK_HORIZON = 2.5   # seconds of bounce-aware prediction rolled out per contact
SENSOR_DT = 1.0 / 60.0        # the sensor's frame interval, for velocity by finite difference

# Prediction uncertainty: obstacles also turn at random (config.ROAM_TURN_RATE),
# which no prediction from the present can foresee, so each predicted
# obstacle is widened by PREDICTION_SPREAD per second of lead time wherever
# a prediction is used (route penalties, the dynamic window's rollouts, the
# ball's landing check) -- the planner keeps clear of everywhere an
# obstacle plausibly *could* be, not just its single most likely path.
# Measured with random turns, bounce-aware prediction's 95th-percentile
# error is 7cm 0.5s ahead, 18cm at 1s, and 41cm at 2s: about 0.2 m/s.
# Off: over 90 random-turn courses every setting tried finished fewer
# (0.2 m/s everywhere: 50/90; route penalties and ball check only: 66-70;
# none: 71-72) and ended more runs with the ball out of bounds -- a wider
# keep-out pushed the robot toward the side lines.
PREDICTION_SPREAD = 0.0       # m of extra radius per second of lead time
DWA_SPREAD = 0.0              # the same, for the dynamic window's rollouts only


@dataclass
class SensedObstacle:
    obs: object       # the Obstacle, for its shape only -- never its motion
    x: float
    y: float
    vx: float
    vy: float
    track: np.ndarray = None   # predicted positions every ROAM_DT ("bounce" prediction)

    def predicted(self, dt):
        if self.track is None:
            return self.x + self.vx * dt, self.y + self.vy * dt
        f = min(max(dt / ROAM_DT, 0.0), len(self.track) - 1.000001)
        i = int(f)
        (x0, y0), (x1, y1) = self.track[i], self.track[i + 1]
        return x0 + (x1 - x0) * (f - i), y0 + (y1 - y0) * (f - i)


class ObstacleSensor:
    """The live planner's only window onto the obstacles: where each one is
    at the current instant, with velocity by finite difference over the
    sensor's last frame (SENSOR_DT) -- a reading one contact apart (~0.3s)
    blurred the velocity across any bounce in between. Static obstacles are known from the map;
    moving ones are read through here, so nothing downstream can peek at
    a roaming obstacle's future."""

    def __init__(self, obstacles):
        self.obstacles = obstacles

    def sense(self, t):
        out = []
        for o in self.obstacles:
            x, y = o.position_at(t)
            vx = vy = 0.0
            if o.moving and t >= SENSOR_DT:
                x0, y0 = o.position_at(t - SENSOR_DT)   # the previous frame: already observed, not the future
                vx, vy = (x - x0) / SENSOR_DT, (y - y0) / SENSOR_DT
            out.append(SensedObstacle(o, x, y, vx, vy))
        if PREDICTION == "bounce":
            movers = [s for s in out if s.obs.moving]
            statics = [s.obs for s in out if not s.obs.moving]
            radii = [s.obs.radius for s in movers]
            pos = [np.array([s.x, s.y]) for s in movers]
            vel = [np.array([s.vx, s.vy]) for s in movers]
            tracks = [[p.copy()] for p in pos]
            for _ in range(int(PREDICT_TRACK_HORIZON / ROAM_DT)):
                pos = roaming_step(radii, pos, vel, statics, ROAM_DT)
                for tr, p in zip(tracks, pos):
                    tr.append(p.copy())
            for s, tr in zip(movers, tracks):
                s.track = np.array(tr)
        return out


class Grid:
    def __init__(self):
        self.nx = int(round((GRID_X1 - GRID_X0) / GRID_RES)) + 1
        self.ny = int(round(2 * COURSE_HALF_WIDTH / GRID_RES)) + 1

    def cell(self, x, y):
        i = int(round((x - GRID_X0) / GRID_RES))
        j = int(round((y - GRID_Y0) / GRID_RES))
        return min(max(i, 0), self.nx - 1), min(max(j, 0), self.ny - 1)

    def center(self, c):
        return GRID_X0 + c[0] * GRID_RES, GRID_Y0 + c[1] * GRID_RES

    def cells_near(self, cx, cy, r):
        i0, j0 = self.cell(cx - r, cy - r)
        i1, j1 = self.cell(cx + r, cy + r)
        return ((i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1))


def body_radius():
    return ROBOT_RADIUS + PLANNING_MARGIN


def obstacle_cells(grid, obs, cx, cy, spread=0.0):
    """Grid cells where the robot body would overlap obs centered at (cx, cy),
    widened by `spread` for prediction uncertainty."""
    r = body_radius() + spread
    for c in grid.cells_near(cx, cy, obs.reach + r):
        x, y = grid.center(c)
        if shape_blocks(obs, cx, cy, x, y, r):
            yield c


class PredictedObstacle:
    """A sensed moving obstacle as the hybrid A* search sees it: same
    shape, but position_at is the sensor's prediction from "now" (t=0)
    instead of the obstacle's real future motion."""
    moving = True
    amplitude = 0.0

    def __init__(self, s):
        o = s.obs
        self.kind, self.hx, self.hy, self.reach = o.kind, o.hx, o.hy, o.reach
        self.cos_nyaw, self.sin_nyaw, self.radius = o.cos_nyaw, o.sin_nyaw, o.radius
        self.x, self.y, self.sensed = s.x, s.y, s
        self.peak_speed = math.hypot(s.vx, s.vy)

    def position_at(self, t):
        return self.sensed.predicted(t)


@dataclass
class ContactPlan:
    paddle_xy: np.ndarray   # where the ball should land for the next contact
    phi: float              # arm angle to arrive with
    speed: float            # horizontal push speed
    path: list              # current D* Lite body path, world coords (for drawing)


class LivePlanner:
    def __init__(self, obstacles, start_body_xy, search="dwa"):
        self.search = search
        self.sensor = ObstacleSensor(obstacles)
        self.grid = Grid()
        self.static_blocked = set()
        for c in ((i, j) for i in range(self.grid.nx) for j in range(self.grid.ny)):
            if abs(self.grid.center(c)[1]) > COURSE_HALF_WIDTH - BASE_XY_HALF:
                self.static_blocked.add(c)
        for o in obstacles:
            if not o.moving:
                self.static_blocked.update(obstacle_cells(self.grid, o, o.x, o.y))
        # Goal cells start half a cell past the finish line: a body at
        # x=5.96 rounds into the cell centered on 6.0, and if that counted
        # as arrived the robot would stop there forever, 4cm short of
        # actually finishing.
        goals = [(i, j) for i in range(self.grid.nx) for j in range(self.grid.ny)
                 if self.grid.center((i, j))[0] >= COURSE_LENGTH + GRID_RES / 2
                 and (i, j) not in self.static_blocked]
        self.dstar = DStarLite(self.grid.nx, self.grid.ny, goals,
                               self.grid.cell(*start_body_xy), self.static_blocked)
        self.heading = 0.0
        self.replans = 0
        self.no_path_contacts = 0
        # Per contact, what ruled out the arm angle the robot would have
        # preferred (closest to its current angle, ball beside the body):
        # "turn_rate", "ball_speed", "obstacle", or "none" if it was free to
        # take it. "local_avoid" counts contacts where the dynamic window
        # passed up the velocity making the most progress for safety.
        self.binding = Counter()
        self.cruise_log = []   # (clearance, cruise speed) per contact
        self.last_v = np.zeros(2)   # body velocity chosen at the previous contact

    def _occupancy(self, sensed, hard=True):
        """(blocked cells, {cell: penalty}) from the sensed obstacles. With
        hard=False nothing moving is blocked: the near-term sweep gets
        CURRENT_PENALTY instead."""
        blocked = set(self.static_blocked)
        penalty = {}
        for s in sensed:
            if not s.obs.moving:
                continue
            for k in range(int(PREDICT_HORIZON / PREDICT_DT) + 1):
                lead = k * PREDICT_DT
                cells = obstacle_cells(self.grid, s.obs, *s.predicted(lead), PREDICTION_SPREAD * lead)
                if lead <= HARD_HORIZON:
                    if hard:
                        blocked.update(cells)
                    else:
                        for c in cells:
                            penalty[c] = max(penalty.get(c, 0.0), CURRENT_PENALTY)
                else:
                    w = PREDICT_PENALTY * (PREDICT_HORIZON - lead) / (PREDICT_HORIZON - HARD_HORIZON)
                    for c in cells:
                        if w > penalty.get(c, 0.0):
                            penalty[c] = w
        return blocked, {c: w for c, w in penalty.items() if c not in blocked}

    def _evade_direction(self, sensed, body_xy):
        """Unit vector away from the moving obstacle that will be nearest
        over the hard horizon, weighted toward the finish when it's not
        directly in the way; None if nothing moving is close."""
        nearest, nearest_d = None, math.inf
        for s in sensed:
            if not s.obs.moving:
                continue
            for k in range(int(HARD_HORIZON / PREDICT_DT) + 1):
                cx, cy = s.predicted(k * PREDICT_DT)
                d = math.hypot(body_xy[0] - cx, body_xy[1] - cy) - s.obs.radius
                if d < nearest_d:
                    nearest, nearest_d = (cx, cy), d
        if nearest is None:
            return None
        away = np.array(body_xy) - nearest
        n = np.linalg.norm(away)
        return away / n if n > 1e-9 else np.array([-1.0, 0.0])

    def _ball_clear(self, sensed, ball_xy, target_xy, flight_time):
        """Ball clear of every obstacle along its straight flight from
        ball_xy to target_xy, and sitting at target_xy for a while after,
        against each obstacle's predicted position at those times."""
        if not ball_in_bounds(*target_xy, LANDING_MARGIN):
            return False
        r = BALL_CLEARANCE + PLANNING_MARGIN
        samples = [(f * flight_time, ball_xy + (target_xy - ball_xy) * f) for f in (0.5, 1.0)]
        samples += [(flight_time + k * PREDICT_DT, target_xy)
                    for k in range(1, int(BALL_PREDICT_HORIZON / PREDICT_DT) + 1)]
        for s in sensed:
            for dt, (x, y) in samples:
                if shape_blocks(s.obs, *s.predicted(dt), x, y, r + PREDICTION_SPREAD * dt):
                    return False
        return True

    def _clearance(self, sensed, xy):
        """Rough distance from xy to the nearest obstacle's bounding circle,
        for picking the least-bad arm angle when none is fully clear."""
        return min(math.hypot(xy[0] - s.x, xy[1] - s.y) - s.obs.radius for s in sensed)

    def plan_contact(self, t, ball_xy, phi, flight_time):
        """Replans from the current sensed state and returns where the ball
        should land next contact and the arm angle to arrive with."""
        ball_xy = np.asarray(ball_xy, dtype=float)
        body_xy = ball_xy + ARM_LENGTH * np.array([math.cos(phi), math.sin(phi)])
        sensed = self.sensor.sense(t)
        self.replans += 1
        self.ball_xy, self.phi = ball_xy, phi   # for the dynamic window's in-bounds check
        body_target, path = {"dstar": self._step_dstar, "hybrid": self._step_hybrid,
                             "dwa": self._step_dwa}[self.search](sensed, body_xy, flight_time)
        target, new_phi = self._choose_arm(sensed, ball_xy, phi, body_target, flight_time)
        speed = np.linalg.norm(target - ball_xy) / flight_time
        return ContactPlan(target, new_phi, speed, path)

    # --- body step, one per search strategy ----------------------------------
    def _chased(self, sensed, body_xy, travel_dir):
        """True if a moving obstacle behind the robot (relative to
        travel_dir) is closing on it and would reach it within
        CHASE_HORIZON at the current closing speed."""
        for s in sensed:
            if not s.obs.moving:
                continue
            r = np.array([s.x, s.y]) - body_xy
            d = float(np.linalg.norm(r))
            gap = d - s.obs.radius - ROBOT_RADIUS
            if d < 1e-9 or gap > CHASE_RANGE or float(r @ travel_dir) >= 0:
                continue
            closing = -float(r @ (np.array([s.vx, s.vy]) - self.last_v)) / d
            if closing > 1e-3 and gap / closing < CHASE_HORIZON:
                return True
        return False

    def _cruise(self, sensed, body_xy, path):
        """Cruise speed for this contact, from the least free space to a
        static obstacle at the robot body or along the next
        CLEARANCE_LOOKAHEAD of route (planning.clearance_at,
        planning.cruise_speed)."""
        pts, travelled = [tuple(body_xy)], 0.0
        for a, b in zip(path, path[1:]):
            travelled += math.dist(a, b)
            if travelled > CLEARANCE_LOOKAHEAD:
                break
            pts.append(b)
        clearance = min(clearance_at([s.obs for s in sensed], x, y) for x, y in pts)
        cruise = cruise_speed(clearance)
        self.cruise_log.append((clearance, cruise))
        return cruise

    def _lookahead(self, path, body_xy):
        """Point LOOKAHEAD along a grid path (not just its next cell), so the
        8-connected grid's zig-zags smooth out."""
        aim, travelled = path[-1], 0.0
        for a, b in zip(path, path[1:]):
            travelled += math.dist(a, b)
            if travelled >= LOOKAHEAD:
                aim = b
                break
        return np.array(aim)

    def _no_path_step(self, sensed, body_xy, in_danger, step):
        self.no_path_contacts += 1
        away = self._evade_direction(sensed, body_xy) if in_danger else None
        if away is not None:
            # Cornered and something is closing in: back off, as far as a
            # step allows without leaving the course or entering a static
            # obstacle's cells.
            for frac in (1.0, 0.5):
                cand = body_xy + away * step * frac
                if self.grid.cell(*cand) not in self.static_blocked:
                    return cand
        return body_xy   # boxed in: dribble in place and wait for the next reading

    def _step_dstar(self, sensed, body_xy, flight_time):
        """D* Lite alone: moving obstacles' near-term sweep hard-blocked in
        the grid, the rest penalized; follow the path at the cruise speed."""
        blocked, penalty = self._occupancy(sensed)
        start = self.grid.cell(*body_xy)
        # The robot's own cell is never treated as blocked -- if an obstacle
        # is closing in, D* Lite should route it *out*, not report no path.
        in_danger = start in blocked and start not in self.static_blocked
        blocked.discard(start)
        self.dstar.update(start, blocked, penalty)
        cells = self.dstar.path()
        path = [self.grid.center(c) for c in cells] if cells else []
        step = (EVADE_SPEED if in_danger else self._cruise(sensed, body_xy, path)) * flight_time
        if len(path) < 2:
            return self._no_path_step(sensed, body_xy, in_danger, step), path
        d = self._lookahead(path, body_xy) - body_xy
        dist = np.linalg.norm(d)
        if dist < 1e-9:
            return body_xy, path
        self.heading = math.atan2(d[1], d[0])
        return body_xy + d / dist * min(step, dist), path

    def _step_hybrid(self, sensed, body_xy, flight_time):
        """planning.hybrid_astar's (x, y, heading, time) lattice search with
        wait edges, rerun from scratch at every contact against obstacles
        extrapolated at their sensed velocity (PredictedObstacle) -- the
        offline planner's search, made live."""
        obstacles = [PredictedObstacle(s) if s.obs.moving else s.obs for s in sensed]
        theta = min(max(self.heading, -THETA_MAX), THETA_MAX)
        path = hybrid_astar(obstacles, (float(body_xy[0]), float(body_xy[1]), theta, 0.0),
                            max_expansions=HYBRID_MAX_EXPANSIONS)
        if not path or len(path) < 2:
            blocked, _ = self._occupancy(sensed)
            in_danger = self.grid.cell(*body_xy) in blocked and self.grid.cell(*body_xy) not in self.static_blocked
            return self._no_path_step(sensed, body_xy, in_danger, EVADE_SPEED * flight_time), []
        # Where the plan has the body one flight from now (it may be a wait).
        (x0, y0, _, t0), (x1, y1, th1, t1) = path[0], path[1]
        f = min(flight_time / (t1 - t0), 1.0)
        if (x1, y1) != (x0, y0):
            self.heading = th1
        return np.array([x0 + (x1 - x0) * f, y0 + (y1 - y0) * f]), [(w[0], w[1]) for w in path]

    def _step_dwa(self, sensed, body_xy, flight_time):
        """Global + local: D* Lite routes over the static map, with moving
        obstacles only as penalties (so a passable course always has a
        route), and a dynamic-window step picks this contact's body
        velocity -- including standing still or backing off -- by rolling
        each candidate forward against the obstacles' predicted motion."""
        _, penalty = self._occupancy(sensed, hard=False)
        start = self.grid.cell(*body_xy)
        self.dstar.update(start, self.static_blocked - {start}, penalty)
        cells = self.dstar.path()
        path = [self.grid.center(c) for c in cells] if cells else []
        aim = self._lookahead(path, body_xy) if len(path) >= 2 else None
        movers = [s for s in sensed if s.obs.moving]
        cruise = self._cruise(sensed, body_xy, path)
        if CHASE_SPEEDUP and aim is not None and self._chased(sensed, body_xy, (aim - body_xy) / max(
                np.linalg.norm(aim - body_xy), 1e-9)):
            cruise = EVADE_SPEED
            self.binding["chased"] += 1
        velocities = dwa_velocities(cruise)

        best, best_score = np.zeros(2), -math.inf
        safest, safest_ttc = np.zeros(2), -math.inf
        # Only while the ball is in bounds: once something has knocked it
        # out, no move "keeps" it in, and filtering on that would leave the
        # robot standing still beside it forever. The arm choice then lands
        # it as far back inside as it can reach each contact instead.
        keep_in = ball_in_bounds(*self.ball_xy)
        for v in velocities:
            heading = math.atan2(v[1], v[0]) if np.linalg.norm(v) > 1e-9 else self.heading
            if keep_in and not self._arm_can_stay_in(body_xy + v * flight_time, heading, flight_time):
                continue   # the ball couldn't stay in bounds behind this move
            ttc, min_gap, hits_static = math.inf, math.inf, False
            for k in range(1, int(DWA_HORIZON / PREDICT_DT) + 1):
                tau = k * PREDICT_DT
                x, y = body_xy + v * tau
                if tau <= DWA_STATIC_HORIZON and self.grid.cell(x, y) in self.static_blocked:
                    hits_static = True
                    break
                for s in movers:
                    cx, cy = s.predicted(tau)
                    gap = math.hypot(x - cx, y - cy) - s.obs.radius - ROBOT_RADIUS - DWA_SPREAD * tau
                    min_gap = min(min_gap, gap)
                    if gap < 0 and tau < ttc:
                        ttc = tau
            if hits_static:
                continue
            if ttc > safest_ttc:
                safest, safest_ttc = v, ttc
            if ttc < DWA_MIN_TTC:
                continue
            nxt = body_xy + v * flight_time
            # Normalized so cruise pace scores 1, and capped there: moving
            # faster than cruise earns nothing for progress alone, so the
            # evasive speed is taken only when safety makes it worth
            # DWA_FAST_COST (uncapped, it became the cruising speed).
            progress = min((np.linalg.norm(aim - body_xy) - np.linalg.norm(aim - nxt)) / (cruise * flight_time),
                           1.0) if aim is not None else 0.0
            score = (progress
                     + DWA_CLEARANCE_WEIGHT * min(min_gap, DWA_SAFE_GAP) / DWA_SAFE_GAP
                     - DWA_TTC_WEIGHT * max(0.0, DWA_HORIZON - ttc) / DWA_HORIZON
                     - DWA_FAST_COST * (np.linalg.norm(v) > cruise + 1e-9))
            if score > best_score:
                best, best_score = v, score
        if best_score == -math.inf:
            self.no_path_contacts += 1
            best = safest   # nothing is safe: take whatever collides latest
        if aim is not None:
            def progress(v):
                return np.linalg.norm(aim - body_xy) - np.linalg.norm(aim - (body_xy + v * flight_time))
            # Avoidance: made less progress than the best move at cruise
            # pace would have (the evasive speed always makes more, so it
            # isn't the yardstick).
            nominal_best = max(progress(v) for v in velocities if np.linalg.norm(v) <= cruise + 1e-9)
            if progress(best) < nominal_best - 1e-6:
                self.binding["local_avoid"] += 1
            if progress(best) < -1e-9:
                self.binding["retreat"] += 1   # backed away from the route to keep clear
        self.last_v = best
        if np.linalg.norm(best) > 1e-9:
            self.heading = math.atan2(best[1], best[0])
        return body_xy + best * flight_time, path

    # --- arm angle (shared) --------------------------------------------------
    def _arm_candidates(self, phi, heading):
        """The current arm angle, plus CHASSIS_OFFSET_CHOICES_DEG off the heading."""
        return [phi] + [heading + math.radians(off) for off in CHASSIS_OFFSET_CHOICES_DEG]

    def _arm_can_stay_in(self, body_target, heading, flight_time):
        """Whether some arm angle the chassis can turn to this contact puts
        the ball at a reachable, in-bounds landing spot for this body
        target -- obstacles aside, which the arm choice handles."""
        reach = BALL_SPEED_MAX * flight_time
        max_turn = BASE_TURN_RATE_MAX * flight_time
        for cand in self._arm_candidates(self.phi, heading):
            if abs(angle_diff_signed(cand, self.phi)) > max_turn + 1e-9:
                continue
            target = paddle_xy(*body_target, cand)
            if ball_in_bounds(*target, LANDING_MARGIN) and math.dist(target, self.ball_xy) <= reach + 1e-9:
                return True
        return False

    def _choose_arm(self, sensed, ball_xy, phi, body_target, flight_time):
        """The candidate arm angle closest to the current one (with a mild
        preference for the ball beside the body) that the chassis can turn
        to (BASE_TURN_RATE_MAX), the ball can reach (BALL_SPEED_MAX), and
        that keeps the ball clear of the predicted obstacles. Also logs
        which of those ruled out the angle preferred with no constraints."""
        reach = BALL_SPEED_MAX * flight_time
        max_turn = BASE_TURN_RATE_MAX * flight_time
        threat = None   # unit vector from the body toward the nearest close moving obstacle
        if SHIELD_WEIGHT > 0:
            nearest = SHIELD_RANGE
            for s in sensed:
                if s.obs.moving:
                    r = np.array([s.x, s.y]) - body_target
                    gap = float(np.linalg.norm(r)) - s.obs.radius - ROBOT_RADIUS
                    if gap < nearest:
                        nearest, threat = gap, r / max(float(np.linalg.norm(r)), 1e-9)
        best, best_score = None, math.inf
        preferred_violations, preferred_score = [], math.inf
        fallback, fallback_clear = None, -math.inf
        least_out, least_out_by = None, math.inf   # last resort: nothing reachable is in bounds
        for cand in self._arm_candidates(phi, self.heading):
            target = np.array(paddle_xy(*body_target, cand))
            side = min(abs(angle_diff_signed(cand, self.heading + s)) for s in (math.pi / 2, -math.pi / 2))
            score = abs(angle_diff_signed(cand, phi)) + ARM_SIDE_WEIGHT * side
            wall_gap = COURSE_HALF_WIDTH - BALL_RADIUS - LANDING_MARGIN - abs(target[1])
            score += WALL_WEIGHT * min(max((WALL_MARGIN - wall_gap) / WALL_MARGIN, 0.0), 1.0)
            if threat is not None:
                ball_dir = -np.array([math.cos(cand), math.sin(cand)])   # body -> ball
                score += SHIELD_WEIGHT * max(0.0, float(ball_dir @ threat))
            violations = []
            if abs(angle_diff_signed(cand, phi)) > max_turn + 1e-9:
                violations.append("turn_rate")
            if np.linalg.norm(target - ball_xy) > reach + 1e-9:
                violations.append("ball_speed")
            kinematic_ok = not violations
            in_bounds = ball_in_bounds(*target, LANDING_MARGIN)
            if not in_bounds:
                violations.append("bounds")
            elif not self._ball_clear(sensed, ball_xy, target, flight_time):
                violations.append("obstacle")
            if score < preferred_score:
                preferred_violations, preferred_score = violations, score
            if not violations:
                if score < best_score:
                    best, best_score = (target, cand), score
            elif kinematic_ok and in_bounds:
                # Least-bad when nothing is fully clear: most room from the
                # obstacles, less any shortfall from the side-line buffer --
                # pure obstacle clearance kept choosing spots right at the
                # line, where the next nudge put the ball out.
                clear = self._clearance(sensed, target) - max(WALL_MARGIN - wall_gap, 0.0)
                if clear > fallback_clear:
                    fallback, fallback_clear = (target, cand), clear
            elif kinematic_ok:
                out_by = max(abs(target[1]) - (COURSE_HALF_WIDTH - BALL_RADIUS - LANDING_MARGIN),
                             -START_SLACK - target[0], 0.0)
                if out_by < least_out_by:
                    least_out, least_out_by = (target, cand), out_by
        for v in preferred_violations or ["none"]:
            self.binding[v] += 1
        if not (best or fallback):
            self.binding["forced_out_of_bounds"] += 1
        return best or fallback or least_out or (ball_xy, phi)
