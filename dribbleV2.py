"""
Court Vision — Objective 2 scaffold, v2
A mobile robot (chassis + rotating arm + paddle) dribbles a ball while
translating and turning down a lane. Structured so later objectives extend
it without a rewrite:
  - Robot: chassis/arm/paddle kinematics + rendering.
  - BallisticModel: nominal flight-time prediction for the vertical bounce.
  - time_to_go(): base guidance -- swap for an obstacle-aware planner (Obj. 3).
  - heading_at(): scripted heading profile -- swap for a path follower (Obj. 3/4).
  - plan_next_contact(): where/when the ball lands next -- Obj. 4's redirect
    planner extends this to pick an alternate push_dir under obstacles.
  - ContactEvent log: extend with fields (redirected, obstacle_density, ...)
    for Obj. 5 metrics without touching call sites.

Nominal, not sensed: FLOOR_E/V_PUSH/HAND_HEIGHT are assumed exact and the
prediction re-anchors from real ball state at every contact, leaning on the
paddle's footprint as slack for the remaining one-cycle error. Obj. 4's
onboard re-estimation replaces this closed-form guess later.

Run:
    pip install numpy pybullet
    python dribbleV2.py
"""

import math
import time
from dataclasses import dataclass

import numpy as np
import pybullet as p
import pybullet_data

DT = 1.0 / 240.0

# --- Tunable parameters -----------------------------------------------------
FLOOR_E = 0.85          # floor coefficient of restitution
HAND_HEIGHT = 0.6       # hand contact-plane height above the floor
HAND_E = 0.8            # effective ball-hand coefficient of restitution
V_PUSH = 3.0            # downward push speed at contact, m/s
LANE_VX = 0.15          # forward drift speed while dribbling, m/s

TURN_START_T = 8.0      # scripted-turn demo: straight -> turn -> straight
TURN_DURATION = 3.0
TURN_ANGLE = math.pi / 2

BALL_RADIUS = 0.12
BALL_MASS = 0.6
PADDLE_HALF = [0.15, 0.15, 0.02]
BASE_XY_HALF = 0.2      # chassis half-width in x/y
BASE_Y = 0.45           # arm length: how far the chassis rides from the ball's line
ARM_HALF = 0.04         # arm thickness

G = 9.81
CONTACT_Z = HAND_HEIGHT - BALL_RADIUS   # ball-center height at the hand contact plane
FLOOR_Z = BALL_RADIUS                    # ball-center height at the floor
PADDLE_CENTER_Z = HAND_HEIGHT + PADDLE_HALF[2]


def heading_at(time_s):
    """Scripted open-loop heading (rad). Swap for a path follower once
    Obj. 3 adds a route to track."""
    if time_s <= TURN_START_T:
        return 0.0
    if time_s >= TURN_START_T + TURN_DURATION:
        return TURN_ANGLE
    return TURN_ANGLE * (time_s - TURN_START_T) / TURN_DURATION


class BallisticModel:
    """Nominal vertical-bounce timing (fall to floor, rebound to hand
    height). Independent of push direction/speed choice, so Obj. 4's
    redirected hits can reuse it with a different v_push."""

    def __init__(self, v_push=V_PUSH, floor_e=FLOOR_E, contact_z=CONTACT_Z, floor_z=FLOOR_Z, g=G):
        drop = contact_z - floor_z
        v_impact = math.sqrt(v_push ** 2 + 2 * g * drop)
        t1 = (v_impact - v_push) / g
        v_rebound = floor_e * v_impact
        t2 = (v_rebound - math.sqrt(v_rebound ** 2 - 2 * g * drop)) / g
        self.flight_time = t1 + t2


def mirror_law_push(vz_in, hand_e=HAND_E, v_push=V_PUSH):
    """Outgoing vertical speed at contact. Always -v_push regardless of
    vz_in (the hand_e*vz terms cancel) -- kept in mirror-law form so a real
    force/torque paddle can drop in later without changing the call site."""
    v_paddle_cmd = (-v_push + hand_e * vz_in) / (1 + hand_e)
    return v_paddle_cmd * (1 + hand_e) - hand_e * vz_in


def time_to_go(current_xy, target_xy, remaining_time, fallback_velocity):
    """Straight-line guidance: velocity that closes the gap by
    `remaining_time`. Obj. 3 swaps this for an obstacle-aware planner with
    the same signature."""
    if remaining_time > DT:
        return (target_xy - current_xy) / remaining_time
    return fallback_velocity


@dataclass
class ContactTarget:
    """Where/when the paddle must meet the ball next."""
    xy: np.ndarray
    t: float
    push_dir: np.ndarray


def plan_next_contact(ball_xy, t_now, heading, bounce: BallisticModel, lane_speed=LANE_VX):
    """Nominal prediction: push along the current heading, land after one
    bounce cycle. Obj. 4 hooks in here to check `xy` against an obstacle
    map and choose a different push_dir when blocked."""
    push_dir = np.array([math.cos(heading), math.sin(heading)])
    xy = np.array(ball_xy) + lane_speed * bounce.flight_time * push_dir
    return ContactTarget(xy=xy, t=t_now + bounce.flight_time, push_dir=push_dir)


@dataclass
class ContactEvent:
    """One dribble cycle's record. Add fields here (redirected,
    obstacle_density, ...) for Obj. 5 metrics without touching call sites."""
    t: float
    x: float
    y: float
    heading_deg: float


class Robot:
    """Kinematic mobile base: chassis + arm + paddle. Owns forward
    kinematics and rendering only; trajectory/heading planning lives
    outside so planners can be swapped without touching this class."""

    def __init__(self, base_y=BASE_Y, base_xy_half=BASE_XY_HALF, arm_half=ARM_HALF):
        self.base_y = base_y
        self.arm_reach = base_y - base_xy_half
        base_half = [base_xy_half, base_xy_half, PADDLE_CENTER_Z / 2]  # chassis top == arm/paddle height
        self.base_z = base_half[2]

        base_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=base_half, rgbaColor=[0.2, 0.7, 0.3, 1])
        self.base_id = p.createMultiBody(0, -1, base_vis, [0, base_y, self.base_z])

        arm_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[arm_half, self.arm_reach / 2, arm_half],
                                       rgbaColor=[0.2, 0.7, 0.3, 1])
        self.arm_id = p.createMultiBody(0, -1, arm_vis, [0, self.arm_reach / 2, PADDLE_CENTER_Z])

        # Base/arm are visual-only (no collision shape): position stand-ins,
        # not physical obstacles for the robot's own ball. Only the paddle
        # needs real collision, for contact detection.
        paddle_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=PADDLE_HALF)
        paddle_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=PADDLE_HALF, rgbaColor=[0.2, 0.2, 0.8, 1])
        self.paddle_id = p.createMultiBody(0, paddle_col, paddle_vis, [0, 0, PADDLE_CENTER_Z])
        p.changeDynamics(self.paddle_id, -1, restitution=0.0)

        self.base_xy = np.array([0.0, base_y])

    def arm_offset(self, heading):
        """World-frame vector from paddle to base at the given heading."""
        return np.array([-self.base_y * math.sin(heading), self.base_y * math.cos(heading)])

    def paddle_xy(self, heading):
        return self.base_xy - self.arm_offset(heading)

    def set_pose(self, base_xy, heading, paddle_z=PADDLE_CENTER_Z):
        """Push a new base pose through forward kinematics to base/arm/paddle."""
        self.base_xy = base_xy
        quat = p.getQuaternionFromEuler([0, 0, heading])
        paddle_xy = self.paddle_xy(heading)
        arm_mid = paddle_xy + (self.arm_reach / 2) * (self.arm_offset(heading) / self.base_y)

        p.resetBasePositionAndOrientation(self.base_id, [*base_xy, self.base_z], quat)
        p.resetBasePositionAndOrientation(self.arm_id, [*arm_mid, PADDLE_CENTER_Z], quat)
        p.resetBasePositionAndOrientation(self.paddle_id, [*paddle_xy, paddle_z], quat)
        return paddle_xy

    def touches(self, other_id):
        return bool(p.getContactPoints(self.paddle_id, other_id))


def spawn_ball(xy0=(0.0, 0.0), vxy0=(LANE_VX, 0.0)):
    """Seeded at the contact plane with the push velocity already applied,
    as if the hand just pushed it (avoids a from-rest first drop that never
    reaches the paddle). Reusable per-trial for Obj. 5 evaluation sweeps."""
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_RADIUS)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=BALL_RADIUS, rgbaColor=[1, 0.4, 0, 1])
    body = p.createMultiBody(BALL_MASS, col, vis, [xy0[0], xy0[1], CONTACT_Z - 0.001])
    p.changeDynamics(body, -1, restitution=FLOOR_E)
    p.resetBaseVelocity(body, linearVelocity=[vxy0[0], vxy0[1], -V_PUSH])
    return body


# --- World setup -------------------------------------------------------------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(DT)

plane = p.loadURDF("plane.urdf")
p.changeDynamics(plane, -1, restitution=FLOOR_E)

ball = spawn_ball()
robot = Robot()
bounce = BallisticModel()

# --- Simulation loop -----------------------------------------------------
PUSH_DEPTH, PUSH_STEPS = 0.05, 12
push_timer = 0
contact_cooldown = 0
target = plan_next_contact([0.0, 0.0], 0.0, 0.0, bounce)
contact_log = []
t = 0.0

for step in range(20000):
    p.stepSimulation()
    t += DT

    ball_pos, _ = p.getBasePositionAndOrientation(ball)
    ball_vel, _ = p.getBaseVelocity(ball)
    vz = ball_vel[2]

    heading = heading_at(t)
    # Look ahead to the heading at arrival (valid since the plan is
    # open-loop) so the base target accounts for the arm rotating with it.
    base_target_xy = target.xy + robot.arm_offset(heading_at(target.t))
    fallback = LANE_VX * np.array([math.cos(heading), math.sin(heading)])
    new_base_xy = robot.base_xy + time_to_go(robot.base_xy, base_target_xy, target.t - t, fallback) * DT

    paddle_z = PADDLE_CENTER_Z - PUSH_DEPTH * (push_timer / PUSH_STEPS) if push_timer > 0 else PADDLE_CENTER_Z
    push_timer = max(push_timer - 1, 0)
    robot.set_pose(new_base_xy, heading, paddle_z)

    # No vz>0 gate: mirror_law_push is an identity in vz, and gating on its
    # sign can silently drop the contact if PyBullet's own collision
    # response zeroes vz before this code sees it.
    if robot.touches(ball) and contact_cooldown <= 0:
        v_new_z = mirror_law_push(vz)
        target = plan_next_contact(ball_pos[:2], t, heading, bounce)
        p.resetBaseVelocity(ball, linearVelocity=[*(LANE_VX * target.push_dir), v_new_z])
        contact_cooldown, push_timer = 20, PUSH_STEPS

        contact_log.append(ContactEvent(t, ball_pos[0], ball_pos[1], math.degrees(heading)))
        print(f"contact #{len(contact_log):3d}  t={t:6.2f}s  xy=({ball_pos[0]:.3f}, {ball_pos[1]:.3f})m  "
              f"heading={math.degrees(heading):5.1f}deg -> next xy=({target.xy[0]:.3f}, {target.xy[1]:.3f})m "
              f"@ t={target.t:.2f}s")
    else:
        contact_cooldown -= 1

    time.sleep(DT)

p.disconnect()

print(f"\nNominal flight time per cycle: {bounce.flight_time:.3f}s")
print(f"Contacts recorded: {len(contact_log)}")
if contact_log:
    c0, c1 = contact_log[0], contact_log[-1]
    dist = math.hypot(c1.x - c0.x, c1.y - c0.y)
    print(f"Net distance traveled: {dist:.2f}m over {c1.t - c0.t:.1f}s "
          f"(heading {c0.heading_deg:.0f}->{c1.heading_deg:.0f}deg)")
