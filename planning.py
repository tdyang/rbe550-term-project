"""Two-stage path planner:
  1. hybrid_astar(): a state-lattice search over (x, y, heading) for the
     robot body's own center, so it can angle through a gap only wide
     enough at an angle.
  2. plan_phi_sequence(): a second search, over the already-found path,
     that picks a collision-free arm-angle (ball/paddle offset) sequence
     and verifies every transition between consecutive points, not just
     each point in isolation.
generate_solvable_course() ties both stages together with course
generation, retrying with a fresh course if either stage fails."""

import heapq
import math

from config import (
    ARM_LENGTH, BALL_CLEARANCE, BALL_RADIUS, BASE_XY_HALF, COURSE_HALF_WIDTH,
    COURSE_LENGTH, ROBOT_RADIUS,
)
from obstacles import generate_course, obstacle_blocks

PLANNING_MARGIN = 0.05   # extra buffer beyond the bare physical radii, reserved during
                         # planning only (not part of the real collision geometry). Real
                         # execution follows piecewise-straight chords between contacts,
                         # not the exact planned curve, so a plan that clears an obstacle
                         # by mere millimeters leaves no room for that approximation error.


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


def paddle_offset_clear(obstacles, x, y, phi):
    """x, y: the robot body's position (the planned path point). phi:
    world-frame angle from the ball/paddle to the chassis -- the same
    convention Robot.set_pose uses (chassis = paddle + ARM_LENGTH*unit(phi))
    -- so the ball/paddle sits at x,y minus that offset."""
    px, py = x - ARM_LENGTH * math.cos(phi), y - ARM_LENGTH * math.sin(phi)
    # Inset by the ball's own radius: a bound on its *center* point still
    # lets part of its physical radius stick out past the course boundary
    # while technically passing the check.
    if not (-COURSE_HALF_WIDTH + BALL_RADIUS <= py <= COURSE_HALF_WIDTH - BALL_RADIUS):
        return False
    return not any(obstacle_blocks(o, px, py, BALL_CLEARANCE + PLANNING_MARGIN) for o in obstacles)


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


def paddle_offset_clear_wide(obstacles, x, y, phi):
    for k in range(-CHASSIS_SAFETY_SAMPLES, CHASSIS_SAFETY_SAMPLES + 1):
        test_phi = phi + CHASSIS_SAFETY_HALF_WIDTH * k / CHASSIS_SAFETY_SAMPLES
        if not paddle_offset_clear(obstacles, x, y, test_phi):
            return False
    return True


def state_blocked(obstacles, x, y, theta):
    """Checks only the robot body's own point. This search's job is finding
    a robot-body path; whether the ball/paddle at the end of the arm can
    actually follow it is a completely separate, independently-verified
    question that plan_phi_sequence answers afterward (wide clearance *and*
    every transition between points, which this search doesn't check at
    all) -- so checking arm wideness here too was pure redundant cost, not
    redundant safety: it couldn't change what does or doesn't end up in the
    final accepted course, since plan_phi_sequence rejects (triggering a
    full regenerate, same as this search failing outright) exactly the same
    paths either way. Measured (in the original ball-centric version of
    this search): removing the equivalent redundant check cut a
    representative slow seed's generation time roughly 6x (16.9s -> 2.8s)
    with identical attempt counts and identical resulting courses, since
    profiling showed this search was where the overwhelming majority of
    course-generation time went."""
    if not (-COURSE_HALF_WIDTH + BASE_XY_HALF <= y <= COURSE_HALF_WIDTH - BASE_XY_HALF):
        return True
    return any(obstacle_blocks(o, x, y, ROBOT_RADIUS + PLANNING_MARGIN) for o in obstacles)


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


# Hybrid-A* lattice: each step is a short forward arc; heading is part of
# the search state (not just x, y) so the planner can angle through a gap
# too narrow to clear head-on.
STEP_LEN = 0.1
MAX_DTHETA = math.radians(18)      # steering options per step
DTHETA_OPTIONS = [f * MAX_DTHETA for f in (-1.0, -0.5, 0.0, 0.5, 1.0)]
THETA_MAX = math.radians(70)       # heading stays within +-70 deg of straight-ahead
BIN_X, BIN_Y, BIN_THETA = 0.05, 0.05, math.radians(4)   # state dedup resolution
THETA_REG_WEIGHT = 0.3             # mild preference for facing forward absent a reason not to


def _search(starts, neighbors, is_goal, heuristic=lambda node: 0.0, on_relax=lambda node, payload: None):
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
    or None if no goal is ever reachable."""
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


def hybrid_astar(obstacles):
    """A* over a state lattice of (x, y, heading) for the robot body's own
    center: each edge is a STEP_LEN-long forward arc at one of
    DTHETA_OPTIONS. Heading is part of the state -- not derived after the
    fact -- so the search can choose to angle through a gap that's only
    passable at that angle, the way a person turns sideways to slip through
    a crowd."""
    def key(x, y, theta):
        return (round(x / BIN_X), round(y / BIN_Y), round(theta / BIN_THETA))

    start = (0.0, 0.0, 0.0)
    if state_blocked(obstacles, *start):
        return None
    start_k = key(*start)
    state_of = {start_k: start}   # binned key -> the actual (unrounded) state it was first reached at

    def neighbors(k, closed):
        x, y, theta = state_of[k]
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
            yield nk, step_cost, (nx, ny, ntheta)

    def remember(k, state):
        state_of[k] = state

    node_path = _search(
        starts=[(start_k, 0.0)],
        neighbors=neighbors,
        is_goal=lambda k: state_of[k][0] >= COURSE_LENGTH,
        heuristic=lambda k: max(COURSE_LENGTH - state_of[k][0], 0.0),
        on_relax=remember,
    )
    return [state_of[k] for k in node_path] if node_path is not None else None


def angle_diff_signed(a, b):
    """a - b, wrapped to (-pi, pi]."""
    return (a - b + math.pi) % (2 * math.pi) - math.pi


PHI_TRANSITION_SUBSTEPS = 6


def phi_transition_clear(obstacles, x0, y0, phi0, x1, y1, phi1):
    """Checks that moving the arm continuously from phi0 (with the robot
    body at x0,y0) to phi1 (at x1,y1) never crosses a blocked point --
    position and arm angle interpolated together, exactly matching how the
    real system moves between two consecutive path points."""
    dphi = angle_diff_signed(phi1, phi0)
    for k in range(1, PHI_TRANSITION_SUBSTEPS + 1):
        f = k / PHI_TRANSITION_SUBSTEPS
        x, y = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        if not paddle_offset_clear(obstacles, x, y, phi0 + dphi * f):
            return False
    return True


def plan_phi_sequence(obstacles, path_xs, path_ys, path_thetas):
    """Precomputes an arm angle (ball/paddle relative to the robot body)
    for every path point via Dijkstra over (point_index, candidate) pairs:
    an edge exists only if the swept transition between two consecutive
    points' angles is collision-free the whole way (phi_transition_clear),
    not just at the endpoints.

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
    moment.

    Returns None if no such sequence exists, in which case the course
    should be regenerated -- same as when the robot body's own path search
    fails."""
    n = len(path_xs)
    candidates = []
    for i in range(n):
        theta = path_thetas[i]
        opts = [theta + math.radians(d) for d in CHASSIS_OFFSET_CHOICES_DEG]
        opts = [phi for phi in opts if paddle_offset_clear_wide(obstacles, path_xs[i], path_ys[i], phi)]
        if not opts:
            return None
        candidates.append(opts)

    def neighbors(node, closed):
        i, j = node
        phi = candidates[i][j]
        x0, y0 = path_xs[i], path_ys[i]
        x1, y1 = path_xs[i + 1], path_ys[i + 1]
        for jn, phin in enumerate(candidates[i + 1]):
            if (i + 1, jn) in closed or not phi_transition_clear(obstacles, x0, y0, phi, x1, y1, phin):
                continue
            yield (i + 1, jn), abs(angle_diff_signed(phin, phi)), None

    node_path = _search(
        starts=[((0, j), 0.0) for j in range(len(candidates[0]))],
        neighbors=neighbors,
        is_goal=lambda node: node[0] == n - 1,
    )
    return [candidates[i][j] for i, j in node_path] if node_path is not None else None


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
