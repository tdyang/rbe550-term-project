"""Ballistic bounce timing, the mirror-law push rule, and ball spawning.
Unchanged from v2."""

import math
from dataclasses import dataclass

import numpy as np
import pybullet as p

from config import BALL_MASS, BALL_RADIUS, CONTACT_Z, FLOOR_E, FLOOR_Z, G, HAND_E, LANE_VX, V_PUSH


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
