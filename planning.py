"""Two-stage path planner, now over space *and* time:
  1. hybrid_astar(): a state-lattice search over (x, y, heading, time) for
     the robot body's own center, so it can angle through a gap only wide
     enough at an angle -- and, since a minority of obstacles move on a
     known schedule (Obstacle.position_at), route around where one of them
     will actually be at the time the robot would get there, not just
     where it started.
  2. plan_phi_sequence(): a second search, over the already-found path,
     that picks a collision-free arm-angle (ball/paddle offset) sequence
     and verifies every transition between consecutive points, not just
     each point in isolation -- also time-aware for the same reason.
generate_solvable_course() ties both stages together with course
generation, retrying with a fresh course if either stage fails.

Time enters the search as an extra, exactly-known coordinate rather than
something reacted to live: every lattice edge is a STEP_LEN-long arc
traversed at the cruise speed for the free space around it (cruise_speed:
slower in tight spots, faster in the open), so the time at a path point
is the sum of its edges' durations. Because that's deterministic, moving obstacles can be checked against
their actual position at that exact time (Obstacle.position_at) during
planning itself, proving the whole run collision-free in advance exactly
like the static case -- no live replanning, no guess about where a moving
obstacle will be, and no regression for the static obstacles that still
make up the rest of any course."""

import heapq
import math

from config import (
    ARM_LENGTH, BALL_CLEARANCE, BALL_RADIUS, BALL_SPEED_MAX, BASE_SPEED_MAX,
    BASE_TURN_RATE_MAX, BASE_XY_HALF, CLEARANCE_RANGE, COURSE_HALF_WIDTH, COURSE_LENGTH,
    CRUISE_SPEED_RANGE, LANE_VX, ROBOT_RADIUS,
)
from obstacles import generate_course, obstacle_blocks

PLANNING_MARGIN = 0.05   # extra buffer beyond the bare physical radii, reserved during
                         # planning only (not part of the real collision geometry). Real
                         # execution follows piecewise-straight chords between contacts,
                         # not the exact planned curve, so a plan that clears an obstacle
                         # by mere millimeters leaves no room for that approximation error.
TIME_MARGIN = 0.25       # seconds of timing error, either way, reserved against every
                         # moving obstacle (obstacle_blocks inflates it by its peak speed
                         # times this). The live controller re-syncs the ball to the
                         # plan's clock at every contact (simulation.plan_next_contact),
                         # but can still be up to about one dribble cycle (~0.23s) off
                         # between contacts -- the spatial PLANNING_MARGIN's counterpart
                         # in time.


# Candidate arm angles (ball/paddle relative to the robot body's heading),
# at 15-degree resolution (23 options spanning the full circle, excluding
# straight ahead where the ball would sit right on top of the chassis).
# plan_phi_sequence picks among these via its own cost search; state_blocked
# only needs to know at least one is ever clear. A coarser grid (originally
# just 7 choices at 45-degree spacing) could leave every single candidate
# blocked at once in a tight spot even though a fine-grained clear angle
# genuinely existed nearby. Ordered by distance from +-90 (beside the
# chassis, usually the most open direction) rather than numerically:
# state_blocked's any() stops at the first clear candidate it finds, and
# this ordering is what that search spends the overwhelming majority of
# course-generation time in (profiled), so checking the likely-open
# directions first measurably speeds it up -- the resulting *set* of
# candidates, and everywhere else that uses it, is unaffected by the order.
CHASSIS_OFFSET_CHOICES_DEG = sorted((d for d in range(-180, 180, 15) if d != 0),
                                     key=lambda d: min(abs(d - 90), abs(d + 90)))


# Planned landing spots stay this far inside the lines: real landings
# scatter by a centimeter or so (floor friction, spin), and spots aimed
# exactly at the limit had the ball riding along the side line just over it.
LANDING_MARGIN = 0.05


# How far behind the start line the ball may go. It starts on the line,
# and the arm-angle checks sample a window of angles around each candidate,
# so at the start some of them put the ball a little behind it; a hard
# x >= 0 made every rigid-arm start state fail. Beyond this, though, the
# robot would be dribbling around behind the course.
START_SLACK = 0.25


def ball_in_bounds(x, y, margin=0.0):
    """The ball inside the course: at least `margin` inside the side lines
    (inset by its own radius too -- a bound on its center alone lets half
    the ball stick out), and no more than START_SLACK behind the start
    line."""
    inner = COURSE_HALF_WIDTH - BALL_RADIUS - margin
    return -inner <= y <= inner and x >= -START_SLACK


def paddle_offset_clear(obstacles, x, y, phi, t):
    """x, y: the robot body's position (a planned path point); t: when it
    would be there. phi: world-frame angle from the ball/paddle to the
    chassis -- the same convention Robot.set_pose uses (chassis = paddle +
    ARM_LENGTH*unit(phi)) -- so the ball/paddle sits at x,y minus that
    offset."""
    px, py = paddle_xy(x, y, phi)
    if not ball_in_bounds(px, py, LANDING_MARGIN):
        return False
    return not any(obstacle_blocks(o, px, py, BALL_CLEARANCE + PLANNING_MARGIN, t, TIME_MARGIN) for o in obstacles)


# How wide a safe angular window a candidate needs, not just a single clear
# point. Without this, three-plus obstacles surrounding the chassis can
# pinch the ball/paddle's only viable angle down to a sliver just a couple
# of degrees wide -- clear right now, but gone the instant the robot
# advances a little further, forcing a jump to whatever's clear next (which,
# if the sliver was the only thing near the arm's current side, can be all
# the way on the opposite side, with solid obstacle blocking every angle in
# between). A genuinely wide window can't vanish that fast, so it can always
# be tracked gradually. Matches the 15-degree candidate spacing: a candidate
# is only accepted if its own immediate neighbors would be too.
CHASSIS_SAFETY_HALF_WIDTH = math.radians(15)
CHASSIS_SAFETY_SAMPLES = 2


def paddle_offset_clear_wide(obstacles, x, y, phi, t):
    for k in range(-CHASSIS_SAFETY_SAMPLES, CHASSIS_SAFETY_SAMPLES + 1):
        test_phi = phi + CHASSIS_SAFETY_HALF_WIDTH * k / CHASSIS_SAFETY_SAMPLES
        if not paddle_offset_clear(obstacles, x, y, test_phi, t):
            return False
    return True


def state_blocked(obstacles, x, y, theta, t, arm_offset=None):
    """Checks only the robot body's own point, at the time it would
    actually be there. This search's job is finding a robot-body path;
    whether the ball/paddle at the end of the arm can actually follow it
    is a completely separate, independently-verified question that
    plan_phi_sequence answers afterward (wide clearance *and* every
    transition between points, which this search doesn't check at all) --
    so checking arm wideness here too was pure redundant cost, not
    redundant safety: it couldn't change what does or doesn't end up in the
    final accepted course, since plan_phi_sequence rejects (triggering a
    full regenerate, same as this search failing outright) exactly the same
    paths either way. Measured (in the original ball-centric, time-agnostic
    version of this search): removing the equivalent redundant check cut a
    representative slow seed's generation time roughly 6x (16.9s -> 2.8s)
    with identical attempt counts and identical resulting courses, since
    profiling showed this search was where the overwhelming majority of
    course-generation time went.

    arm_offset (the no-redirect baseline, plan_course(redirect=False)):
    the ball is rigidly fixed at this offset from the heading, so it can
    no longer be placed afterward and must be checked here, as part of the
    robot's own footprint."""
    if not (-COURSE_HALF_WIDTH + BASE_XY_HALF <= y <= COURSE_HALF_WIDTH - BASE_XY_HALF):
        return True
    if any(obstacle_blocks(o, x, y, ROBOT_RADIUS + PLANNING_MARGIN, t, TIME_MARGIN) for o in obstacles):
        return True
    return arm_offset is not None and not paddle_offset_clear_wide(obstacles, x, y, theta + arm_offset, t)


EDGE_SUBSTEPS = 4   # collision-check points along each edge, not just its endpoint


def edge_blocked(obstacles, x0, y0, theta0, t0, x1, y1, theta1, t1, arm_offset=None):
    """Checks several points along the step from (x0,y0,theta0,t0) to
    (x1,y1,theta1,t1), not just the endpoint. STEP_LEN (0.1m) is comparable
    to or larger than some obstacle radii, so endpoint-only checking can
    let an edge tunnel straight through an obstacle that sits between two
    otherwise-clear waypoints -- time is interpolated right along with
    position/heading, since within one step a moving obstacle has moved
    too."""
    for i in range(1, EDGE_SUBSTEPS + 1):
        f = i / EDGE_SUBSTEPS
        if state_blocked(obstacles, x0 + (x1 - x0) * f, y0 + (y1 - y0) * f,
                          theta0 + (theta1 - theta0) * f, t0 + (t1 - t0) * f, arm_offset):
            return True
    return False


# Hybrid-A* lattice: each step is a short forward arc; heading is part of
# the search state (not just x, y) so the planner can angle through a gap
# too narrow to clear head-on.
STEP_LEN = 0.1
MAX_DTHETA = math.radians(18)      # steering options per step
DTHETA_OPTIONS = [f * MAX_DTHETA for f in (-1.0, -0.5, 0.0, 0.5, 1.0)]
THETA_MAX = math.radians(70)       # heading stays within +-70 deg of straight-ahead
BIN_X, BIN_Y, BIN_THETA = 0.05, 0.05, math.radians(4)   # state dedup resolution
THETA_REG_WEIGHT = 0.3             # mild preference for facing forward absent a reason not to

# Time to cross one STEP_LEN arc at the nominal LANE_VX: the length of a
# wait edge, and the time binning of the search's states. Forward edges
# take STEP_LEN / cruise_speed(...) instead.
STEP_TIME = STEP_LEN / LANE_VX


def clearance_at(obstacles, x, y):
    """Free space around the robot body at (x, y): distance to the
    nearest *static* obstacle's bounding circle, less the body's own
    radius (bounding circles make it conservative for boxes). Moving
    obstacles are left out on purpose: a narrow way is fixed geometry,
    and slowing down because something mobile is near only gives it more
    time to reach the robot -- dodging those is the planners' job."""
    c = math.inf
    for o in obstacles:
        if not o.moving:
            c = min(c, math.hypot(x - o.x, y - o.y) - o.radius - ROBOT_RADIUS)
    return c


def cruise_speed(clearance):
    """Travel speed for this much free space: slow and methodical through
    narrow gaps, quicker in the open (CRUISE_SPEED_RANGE over
    CLEARANCE_RANGE, linear between)."""
    (c0, c1), (v0, v1) = CLEARANCE_RANGE, CRUISE_SPEED_RANGE
    return v0 + (v1 - v0) * min(max((clearance - c0) / (c1 - c0), 0.0), 1.0)

# Wait edges: the robot holds its (x, y, heading) and dribbles in place for
# one STEP_TIME, letting a moving obstacle pass instead of only steering
# around it. A wait then a step averages half speed, so this also covers
# "slow down" without any off-grid time. WAIT_COST equals a straight
# step's, so a wait is worth the same as 0.1m of detour: the search still
# prefers moving when moving works.
WAIT_COST = STEP_LEN
# Bounds how much the search may wait in total (time spent beyond what
# pure forward progress at LANE_VX accounts for), so a dead end can't be
# explored at every possible wait time forever.
MAX_WAIT_TIME = 4.0
# Waiting is only offered where a moving obstacle could reach the robot or
# its arm within a few steps -- beside a static wall it can never help,
# and would only multiply every dead end by every possible wait length.
WAIT_LOOKAHEAD = 0.5


def _search(starts, neighbors, is_goal, heuristic=lambda node: 0.0, on_relax=lambda node, payload: None,
            max_expansions=None):
    """Shared Dijkstra/A* core for hybrid_astar and plan_phi_sequence: both
    are priority-queue searches over an implicit graph, differing only in
    what a node means and how starts/neighbors/cost/goal are defined --
    plain Dijkstra when `heuristic` is left at its default (always 0,
    as plan_phi_sequence uses it), A* when it isn't (hybrid_astar's
    distance-to-goal estimate).

    `starts`: iterable of (node, cost). `neighbors(node, closed)`: iterable
    of (next_node, edge_cost, payload) -- `closed` lets a caller skip an
    expensive validity check for a neighbor already known optimal, exactly
    as both searches already did for their own closed-set lookups.
    `on_relax(next_node, payload)` fires exactly when an edge turns out to
    improve next_node's known distance, letting a caller attach bookkeeping
    (hybrid_astar's state_of) only to the best-known path to a node, not to
    every edge that happens to reach it.

    Returns the node sequence from whichever start reaches a goal first,
    or None if no goal is ever reachable (or none is found within
    max_expansions node expansions, when given)."""
    dist, prev, closed = {}, {}, set()
    pq = []
    for node, cost in starts:
        dist[node] = cost
        heapq.heappush(pq, (cost + heuristic(node), node))

    while pq:
        _, node = heapq.heappop(pq)
        if node in closed:
            continue
        closed.add(node)
        if max_expansions is not None and len(closed) > max_expansions:
            return None
        if is_goal(node):
            path = [node]
            while path[-1] in prev:
                path.append(prev[path[-1]])
            path.reverse()
            return path
        for nxt, edge_cost, payload in neighbors(node, closed):
            nd = dist[node] + edge_cost
            if nd < dist.get(nxt, math.inf):
                dist[nxt] = nd
                prev[nxt] = node
                on_relax(nxt, payload)
                heapq.heappush(pq, (nd + heuristic(nxt), nxt))
    return None


def hybrid_astar(obstacles, start=(0.0, 0.0, 0.0, 0.0), max_expansions=None, arm_offset=None):
    """A* over a state lattice of (x, y, heading, time) for the robot
    body's own center: each edge is a STEP_LEN-long forward arc at one of
    DTHETA_OPTIONS, advancing time by STEP_TIME. Heading is part of the
    state -- not derived after the fact -- so the search can choose to
    angle through a gap that's only passable at that angle, the way a
    person turns sideways to slip through a crowd. Time is part of the
    state too, so two paths that reach the same (x,y,heading) at different
    times are genuinely different states: a moving obstacle may block one
    and not the other. Besides the forward arcs, a node near a moving
    obstacle may also wait in place for a STEP_TIME (see WAIT_COST).

    `start` is (x, y, heading, t); the live planner (live_planning.py)
    replans from the robot's current state with t=0 meaning "now", against
    obstacles whose position_at extrapolates what it sensed."""
    def key(x, y, theta, t):
        return (round(x / BIN_X), round(y / BIN_Y), round(theta / BIN_THETA), round(t / STEP_TIME))

    if state_blocked(obstacles, *start, arm_offset):
        return None
    start_k = key(*start)
    state_of = {start_k: start}   # binned key -> the actual (unrounded) state it was first reached at
    moving = [o for o in obstacles if o.moving]

    def wait_useful(x, y):
        return any(math.hypot(x - o.x, y - o.y)
                   <= o.reach + o.amplitude + ROBOT_RADIUS + ARM_LENGTH + WAIT_LOOKAHEAD
                   for o in moving)

    def neighbors(k, closed):
        x, y, theta, t = state_of[k]
        nt = t + STEP_LEN / cruise_speed(clearance_at(obstacles, x, y))
        for dtheta in DTHETA_OPTIONS:
            ntheta = theta + dtheta
            if abs(ntheta) > THETA_MAX:
                continue
            heading_mid = theta + dtheta / 2
            nx, ny = x + STEP_LEN * math.cos(heading_mid), y + STEP_LEN * math.sin(heading_mid)
            if edge_blocked(obstacles, x, y, theta, t, nx, ny, ntheta, nt, arm_offset):
                continue
            # A fixed arm swings the ball through every heading change; turns
            # it can't follow within the speed limits are ruled out here,
            # since no later arm choice can absorb them.
            if arm_offset is not None and not phi_transition_trackable(
                    x, y, theta + arm_offset, t, nx, ny, ntheta + arm_offset, nt):
                continue
            nk = key(nx, ny, ntheta, nt)
            if nk in closed:
                continue
            step_cost = STEP_LEN * (1.0 + THETA_REG_WEIGHT * abs(ntheta) / THETA_MAX)
            yield nk, step_cost, (nx, ny, ntheta, nt)

        # Waiting is time beyond what even the slowest cruise would take
        # to make this much progress.
        wt = t + STEP_TIME
        if wt - start[3] - (x - start[0]) / CRUISE_SPEED_RANGE[0] <= MAX_WAIT_TIME and wait_useful(x, y):
            nk = key(x, y, theta, wt)
            if nk not in closed and not edge_blocked(obstacles, x, y, theta, t, x, y, theta, wt, arm_offset):
                # Same heading regularization as a step, so a wait isn't a
                # cheap way to sit at a steep heading.
                yield nk, WAIT_COST * (1.0 + THETA_REG_WEIGHT * abs(theta) / THETA_MAX), (x, y, theta, wt)

    def remember(k, state):
        state_of[k] = state

    node_path = _search(
        starts=[(start_k, 0.0)],
        neighbors=neighbors,
        is_goal=lambda k: state_of[k][0] >= COURSE_LENGTH,
        heuristic=lambda k: max(COURSE_LENGTH - state_of[k][0], 0.0),
        on_relax=remember,
        max_expansions=max_expansions,
    )
    return [state_of[k] for k in node_path] if node_path is not None else None


def angle_diff_signed(a, b):
    """a - b, wrapped to (-pi, pi]."""
    return (a - b + math.pi) % (2 * math.pi) - math.pi


PHI_TRANSITION_SUBSTEPS = 6


def phi_transition_clear(obstacles, x0, y0, phi0, t0, x1, y1, phi1, t1):
    """Checks that moving the arm continuously from phi0 (with the robot
    body at x0,y0,t0) to phi1 (at x1,y1,t1) never crosses a blocked point
    -- position, arm angle, and time interpolated together, exactly
    matching how the real system moves between two consecutive path
    points."""
    dphi = angle_diff_signed(phi1, phi0)
    for k in range(1, PHI_TRANSITION_SUBSTEPS + 1):
        f = k / PHI_TRANSITION_SUBSTEPS
        x, y = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        if not paddle_offset_clear(obstacles, x, y, phi0 + dphi * f, t0 + (t1 - t0) * f):
            return False
    return True


def paddle_xy(x, y, phi):
    """Ball/paddle position for a robot-body point and arm angle -- the
    inverse of Robot.set_pose's chassis = paddle + ARM_LENGTH*unit(phi)."""
    return x - ARM_LENGTH * math.cos(phi), y - ARM_LENGTH * math.sin(phi)


def phi_transition_trackable(x0, y0, phi0, t0, x1, y1, phi1, t1):
    """True if the robot can keep to this transition's schedule: the
    chassis may not turn faster than BASE_TURN_RATE_MAX (its yaw *is* phi
    -- chassis and arm are one rigid assembly) or move faster than
    BASE_SPEED_MAX, and the paddle's displacement over it, divided by the
    time the plan allows, must not exceed BALL_SPEED_MAX. Without this, an arm flip from one side
    of the chassis to the other in a single STEP_TIME is a perfectly
    collision-free plan that asks the ball to sweep a meter sideways in
    0.4s -- the live controller falls behind schedule, and behind schedule
    is exactly where moving obstacles are no longer where they were
    checked."""
    dt = t1 - t0
    if abs(angle_diff_signed(phi1, phi0)) > BASE_TURN_RATE_MAX * dt:
        return False
    if math.hypot(x1 - x0, y1 - y0) > BASE_SPEED_MAX * dt:
        return False
    px0, py0 = paddle_xy(x0, y0, phi0)
    px1, py1 = paddle_xy(x1, y1, phi1)
    return math.hypot(px1 - px0, py1 - py0) <= BALL_SPEED_MAX * dt


def plan_phi_sequence(obstacles, path_xs, path_ys, path_thetas, path_ts, offsets_deg=CHASSIS_OFFSET_CHOICES_DEG):
    """Precomputes an arm angle (ball/paddle relative to the robot body)
    for every path point via Dijkstra over (point_index, candidate) pairs:
    an edge exists only if the swept transition between two consecutive
    points' angles is collision-free the whole way (phi_transition_clear),
    not just at the endpoints, and slow enough for the ball to keep to
    schedule (phi_transition_trackable).

    This replaces an earlier design where the arm angle was chosen live,
    frame by frame, reacting to obstacles as they were encountered. That
    approach, however well-tuned (durable-candidate lookahead, wide-margin
    candidates, immediate correction when already unsafe), could not
    actually guarantee zero emergency corrections: it only ever verified
    that a candidate was clear *at a point*, never that the ball/paddle
    could actually get from one point's safe angle to the next's within the
    distance available, so three-plus obstacles narrowing the safe zone
    from multiple sides at once could still isolate the tracked angle into
    a shrinking island with no reachable escape, however far ahead the
    lookahead looked. Planning the whole angle sequence in advance and
    verifying every transition, not just every point, removes that gap
    entirely: if this returns a sequence, the arm can move through it
    exactly as planned with no live safety check ever needed again, the
    same way the robot body's own path is proven collision-free in advance
    rather than hoping a live heuristic finds a way through moment to
    moment. path_ts (from hybrid_astar's time-aware search) lets this stay
    just as true when some of the obstacles are moving.

    Returns None if no such sequence exists, in which case the course
    should be regenerated -- same as when the robot body's own path search
    fails."""
    n = len(path_xs)
    candidates = []
    for i in range(n):
        theta = path_thetas[i]
        opts = [theta + math.radians(d) for d in offsets_deg]
        opts = [phi for phi in opts
                if paddle_offset_clear_wide(obstacles, path_xs[i], path_ys[i], phi, path_ts[i])]
        if not opts:
            return None
        candidates.append(opts)

    def neighbors(node, closed):
        i, j = node
        phi = candidates[i][j]
        x0, y0, t0 = path_xs[i], path_ys[i], path_ts[i]
        x1, y1, t1 = path_xs[i + 1], path_ys[i + 1], path_ts[i + 1]
        for jn, phin in enumerate(candidates[i + 1]):
            if (i + 1, jn) in closed or not phi_transition_trackable(x0, y0, phi, t0, x1, y1, phin, t1):
                continue
            if not phi_transition_clear(obstacles, x0, y0, phi, t0, x1, y1, phin, t1):
                continue
            yield (i + 1, jn), abs(angle_diff_signed(phin, phi)), None

    node_path = _search(
        starts=[((0, j), 0.0) for j in range(len(candidates[0]))],
        neighbors=neighbors,
        is_goal=lambda node: node[0] == n - 1,
    )
    return [candidates[i][j] for i, j in node_path] if node_path is not None else None


# Node-expansion budget for one offline hybrid A* search. Unbounded, a
# search on a course with no way through exhausts the whole (x, y, heading,
# time) lattice first -- minutes on dense roaming courses -- only to report
# failure; with the budget it gives up and the course is regenerated.
OFFLINE_MAX_EXPANSIONS = 200_000


def plan_course(obstacles, redirect=True, ignore_moving=False, max_expansions=OFFLINE_MAX_EXPANSIONS):
    """Both planning stages on one course. Returns (path, phis, reason):
    reason is "ok", "body" (hybrid A* found no robot-body path), or "arm"
    (a body path, but no arm-angle sequence along it).

    redirect=False is the no-redirect baseline: the ball rides rigidly
    beside the chassis at +-90 degrees -- the nominal landing spot, never
    shifted -- so it's checked as part of the robot's footprint during the
    body search, and the arm search is offered only that one offset. The
    side is chosen once per course (both are tried), the most generous
    version of a robot that can't redirect the ball.

    ignore_moving=True plans as if the moving obstacles weren't there: the
    check the live planners' courses are accepted by, ruling out only
    courses whose static obstacles leave no room for the body and arm."""
    obs = [o for o in obstacles if not o.moving] if ignore_moving else obstacles
    reason = "body"
    for side in ([None] if redirect else [90, -90]):
        offset = None if side is None else math.radians(side)
        path = hybrid_astar(obs, max_expansions=max_expansions, arm_offset=offset)
        if path is None:
            continue
        cols = [[w[k] for w in path] for k in range(4)]
        phis = plan_phi_sequence(obs, *cols, CHASSIS_OFFSET_CHOICES_DEG if side is None else [side])
        if phis is not None:
            return path, phis, "ok"
        reason = "arm"
    return None, None, reason


def generate_solvable_course(rng, max_attempts=50, motion="oscillate", redirect=True, ignore_moving=False):
    for attempt in range(1, max_attempts + 1):
        obstacles = generate_course(rng, motion)
        path, phis, _reason = plan_course(obstacles, redirect, ignore_moving)
        if path is not None:
            return obstacles, path, phis, attempt
    raise RuntimeError(f"no collision-free path found in {max_attempts} random courses")
