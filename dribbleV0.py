"""
Court Vision — Objective 1 scaffold
Stationary ball-paddle dribble using a mirror-law contact controller.

This is a v0: the paddle is moved kinematically (position/velocity are set
directly, not driven by a motor), and contact is handled with a simple
debounce rather than full PyBullet contact/restitution physics. The goal
here is to nail down the mirror-law math and get a visibly periodic dribble
before wiring in real paddle dynamics, camera-based re-estimation, or the
mobile base.

Run:
    pip install pybullet numpy
    python dribble_v0.py
"""

import time

import numpy as np
import pybullet as p
import pybullet_data

# ---------------------------------------------------------------------------
# Sim setup
# ---------------------------------------------------------------------------
DT = 1.0 / 240.0

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(DT)

plane = p.loadURDF("plane.urdf")

# Ball
BALL_RADIUS = 0.12
BALL_MASS = 0.6
ball_col = p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_RADIUS)
ball_vis = p.createVisualShape(p.GEOM_SPHERE, radius=BALL_RADIUS, rgbaColor=[1, 0.4, 0, 1])
ball = p.createMultiBody(BALL_MASS, ball_col, ball_vis, [0, 0, 1.0])
p.changeDynamics(ball, -1, restitution=0.0)  # we apply restitution ourselves via the mirror law

# Paddle (kinematic box, moved by directly resetting its pose/velocity)
PADDLE_HALF = [0.15, 0.15, 0.02]
paddle_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=PADDLE_HALF)
paddle_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=PADDLE_HALF, rgbaColor=[0.2, 0.2, 0.8, 1])
PADDLE_Z0 = 0.3
paddle = p.createMultiBody(0, paddle_col, paddle_vis, [0, 0, PADDLE_Z0])
p.changeDynamics(paddle, -1, restitution=0.0)

# ---------------------------------------------------------------------------
# Mirror-law controller
#
# 1D contact model (Buhler-Koditschek-style): for a paddle moving at v_p and
# an incoming ball velocity v_ball_minus, an effective coefficient of
# restitution e between ball and paddle gives an outgoing ball velocity:
#
#     v_ball_plus - v_p = -e * (v_ball_minus - v_p)
#
# Solving for v_p given a *desired* v_ball_plus (the apex speed we want the
# ball to leave with) is the mirror law:
#
#     v_p = (v_ball_plus_desired + e * v_ball_minus) / (1 + e)
# ---------------------------------------------------------------------------
E_EFF = 0.8            # effective ball-paddle coefficient of restitution
APEX_HEIGHT = 1.0       # desired bounce height above the paddle, meters
V_OUT_DESIRED = np.sqrt(2 * 9.81 * APEX_HEIGHT)

apex_log = []          # (time, height) at each detected apex, for convergence checking
prev_vz = 0.0
contact_cooldown = 0

t = 0.0
for step in range(20000):
    p.stepSimulation()
    t += DT

    ball_pos, _ = p.getBasePositionAndOrientation(ball)
    ball_vel, _ = p.getBaseVelocity(ball)
    vz = ball_vel[2]

    # crude apex detection: velocity sign flips from + to -
    if prev_vz > 0 and vz <= 0:
        apex_log.append((t, ball_pos[2]))
    prev_vz = vz

    contacts = p.getContactPoints(ball, paddle)
    if contacts and contact_cooldown <= 0 and vz < 0:
        v_paddle_cmd = (V_OUT_DESIRED + E_EFF * vz) / (1 + E_EFF)
        v_ball_new = v_paddle_cmd * (1 + E_EFF) - E_EFF * vz  # should equal V_OUT_DESIRED

        p.resetBaseVelocity(ball, linearVelocity=[ball_vel[0], ball_vel[1], v_ball_new])
        contact_cooldown = 20  # steps; debounce multi-frame contact events
    else:
        contact_cooldown -= 1

    time.sleep(DT)

p.disconnect()

print("Apex heights over time (target = {:.3f} m above paddle):".format(APEX_HEIGHT + PADDLE_Z0))
for t_apex, h in apex_log:
    print(f"  t={t_apex:6.2f}s  z={h:.3f}m")