"""Objective 4 (redirect-the-bounce) metrics, and the start of Objective 5's
evaluation pipeline.

The proposal's Objective 4 was a live, reactive layer: predict the ball's
default (straight-continuation) landing point, and if an obstacle blocks
it, compute an alternate hit angle/impulse that shifts the landing point
to an open cell. The two-stage planner built for Objective 3 already
solves the ball's entire path in advance (via the arm-angle sequence,
plan_phi_sequence), so every bounce is redirected by construction, not
reactively patched -- adding a second, live decision layer on top would
just reintroduce the no-guarantee problem Section III of the status
report describes the team abandoning.

This module makes that already-happening redirection explicit and
measurable instead: a "redirect event" is a point along the solved ball
path whose *nominal* landing spot -- what a planner that ignored every
obstacle would have produced, straight down the course centerline at
y=0 -- is itself blocked at the time the ball would nominally arrive
there. Reusing obstacle_blocks (the same primitive the real planner is
built on) means this can never disagree with what the planner itself
treats as blocked, moving obstacles included.
"""

import math

from config import BALL_CLEARANCE, BALL_SPEED_MAX, BASE_SPEED_MAX, BASE_TURN_RATE_MAX, LANE_VX
from obstacles import obstacle_blocks
from planning import angle_diff_signed, paddle_offset_clear_wide, paddle_xy


def nominal_blocked(obstacles, x):
    """True if the naive straight-ahead ball landing point (x, y=0) is
    itself obstacle-blocked at the time a constant-LANE_VX nominal plan
    would reach it (x / LANE_VX -- the same timing assumption the planner
    itself makes, planning.STEP_TIME) -- the exact condition the
    proposal's Objective 4 redirect layer is specified to react to."""
    return any(obstacle_blocks(o, x, 0.0, BALL_CLEARANCE, x / LANE_VX) for o in obstacles)


def redirect_events(obstacles, ball_xs, ball_ys):
    """Points along the ball's own solved path (paddle_xs/paddle_ys in
    simulation.py) where the nominal straight-ahead spot was blocked,
    paired with how far in y the actual plan ended up away from it there
    -- the magnitude of the "alternate landing point" the proposal
    describes. Returns a list of (index, shift) pairs."""
    return [(i, abs(y)) for i, (x, y) in enumerate(zip(ball_xs, ball_ys)) if nominal_blocked(obstacles, x)]


def summarize_redirects(obstacles, ball_xs, ball_ys):
    events = redirect_events(obstacles, ball_xs, ball_ys)
    n = len(ball_xs)
    shifts = [shift for _, shift in events]
    return {
        "n_points": n,
        "n_redirects": len(events),
        "redirect_fraction": len(events) / n if n else 0.0,
        "mean_shift": sum(shifts) / len(shifts) if shifts else 0.0,
        "max_shift": max(shifts) if shifts else 0.0,
    }


BINDING_FRACTION = 0.95   # a limit counts as binding on a transition using >= this share of it


def plan_constraints(obstacles, path, phis):
    """What shaped an offline plan, for the question "is the ball landing on
    obstacles the bottleneck, or how fast the base can get there?".
    Per path point: whether both nominal landing spots (ball beside the
    body, +-90 degrees) were blocked by an obstacle, forcing a redirect.
    Per transition: whether the base turn rate, base speed, or ball speed
    limit was binding (>= BINDING_FRACTION of the limit used)."""
    n = len(path)
    forced = sum(
        1 for (x, y, th, t) in path
        if not any(paddle_offset_clear_wide(obstacles, x, y, th + s, t) for s in (math.pi / 2, -math.pi / 2)))
    turn = base = ball = 0
    for (x0, y0, _, t0), (x1, y1, _, t1), p0, p1 in zip(path, path[1:], phis, phis[1:]):
        dt = t1 - t0
        turn += abs(angle_diff_signed(p1, p0)) >= BINDING_FRACTION * BASE_TURN_RATE_MAX * dt
        base += math.hypot(x1 - x0, y1 - y0) >= BINDING_FRACTION * BASE_SPEED_MAX * dt
        (a, b), (c, d) = paddle_xy(x0, y0, p0), paddle_xy(x1, y1, p1)
        ball += math.hypot(c - a, d - b) >= BINDING_FRACTION * BALL_SPEED_MAX * dt
    m = max(n - 1, 1)
    return {"redirect_forced_frac": forced / n, "turn_rate_binding_frac": turn / m,
            "base_speed_binding_frac": base / m, "ball_speed_binding_frac": ball / m}
