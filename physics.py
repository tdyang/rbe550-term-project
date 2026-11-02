"""Ballistic bounce timing, the energy-consistent paddle impact, and ball
spawning."""

import math
from dataclasses import dataclass

import numpy as np
import pybullet as p

from config import (
    BALL_MASS, BALL_RADIUS, CONTACT_Z, FLOOR_E, FLOOR_Z, G, HAND_E, LANE_VX, PADDLE_TILT_MAX, V_PUSH,
)


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


@dataclass
class Impact:
    v_out: np.ndarray       # ball velocity after the hit
    normal: np.ndarray      # unit impact normal: paddle face -> ball, the impulse's direction
    paddle_speed: float     # paddle velocity along the normal at impact (m/s)
    tilt: float             # paddle face tilt from horizontal (rad)
    work: float             # J done on the ball by the paddle
    loss: float             # J dissipated in the impact (restitution < 1)
    tilt_limited: bool      # the planner's horizontal push was cut to respect PADDLE_TILT_MAX


def paddle_impact(v_in, v_xy, mass=BALL_MASS, e=HAND_E, v_push=V_PUSH, tilt_max=PADDLE_TILT_MAX):
    """The paddle hit as a rigid-body impact, rather than overwriting the
    ball's velocity (which is how earlier versions did it: no impulse, no
    paddle work, and nothing to say where the added energy came from).

    The outgoing velocity is the mirror law's vertical -v_push (whatever
    the incoming speed, so the bounce height stays constant) plus the
    planner's horizontal push v_xy. For a frictionless impact the impulse,
    and so the velocity change dv, lies along the face normal n; a much
    heavier paddle moving at speed u along n gives
        dv = (1 + e) (u - v_in . n) n,
    so the face is tilted to n = dv/|dv| and driven at
        u = |dv| / (1 + e) + v_in . n.
    The paddle's work is the impulse times its own velocity, m |dv| u, and
    the ball's kinetic energy gain falls short of that by the restitution
    loss, (1/2) m (1 - e^2) (u - v_in . n)^2 -- energy is accounted for at
    every contact, which simulation.py checks over whole runs. Tilting the
    face is also literally the proposal's "alternate hit angle" redirect.

    If the push would need more than tilt_max, the horizontal part is cut
    back to what that tilt allows (vertical mirror law kept)."""
    v_in = np.asarray(v_in, dtype=float)
    dv = np.array([v_xy[0], v_xy[1], -v_push]) - v_in
    tilt = math.atan2(math.hypot(dv[0], dv[1]), -dv[2])
    tilt_limited = tilt > tilt_max
    if tilt_limited:
        h = math.hypot(dv[0], dv[1])
        dv[:2] *= -dv[2] * math.tan(tilt_max) / h
        tilt = tilt_max
    dv_mag = float(np.linalg.norm(dv))
    n = dv / dv_mag
    u = dv_mag / (1 + e) + float(v_in @ n)
    v_out = v_in + dv
    work = mass * dv_mag * u
    loss = work - 0.5 * mass * (float(v_out @ v_out) - float(v_in @ v_in))
    return Impact(v_out, n, u, tilt, work, loss, tilt_limited)


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
    impact: Impact = None


def spawn_ball(xy=(0.0, 0.0), vxy=(LANE_VX, 0.0)):
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_RADIUS)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=BALL_RADIUS, rgbaColor=[1, 0.4, 0, 1])
    body = p.createMultiBody(BALL_MASS, col, vis, [*xy, CONTACT_Z - 0.001])
    # PyBullet applies a default linear damping (0.04/s) -- an unmodeled
    # drag that BallisticModel's flight time doesn't include, and that
    # quietly drained ~8% of the paddle's work in the energy ledger.
    p.changeDynamics(body, -1, restitution=FLOOR_E, linearDamping=0.0, angularDamping=0.0)
    p.resetBaseVelocity(body, linearVelocity=[*vxy, -V_PUSH])
    return body
