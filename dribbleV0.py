"""
Court Vision — Objective 1 scaffold, v1 (fixed)
Basketball-style dribble: a "hand" (paddle) above the ball pushes it down;
the ball falls, bounces off the actual floor (passive restitution), and
rises back up to meet the hand again.

Fix from the first version of this script: the ball used to start at rest
and simply drop, so the floor's passive restitution alone had to carry it
all the way back up to hand height on the very first bounce -- it never
had enough energy, so the hand's contact condition never fired and the
bounces quietly decayed to rest. The ball now starts at hand height with
the push velocity already applied, simulating "the hand just pushed it,"
so the cycle has real energy in it from step one.

Apex-height logging is back (as in v0) specifically so a bug like that is
visible in the console the moment it starts happening, instead of only
being visible by watching the GUI decay over several seconds.

Run:
    pip install numpy pybullet   (or pybullet-arm64 on Apple Silicon)
    python dribble_v1.py
"""

import time

import numpy as np
import pybullet as p
import pybullet_data

DT = 1.0 / 240.0

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(DT)

# ---------------------------------------------------------------------------
# Tunable parameters (grouped up front so they're easy to sweep)
# ---------------------------------------------------------------------------
FLOOR_E = 0.85         # floor coefficient of restitution (passive)
HAND_HEIGHT = 0.6      # height of the hand's contact point above the floor
HAND_E = 0.8           # effective ball-hand coefficient of restitution
V_PUSH = 3.0           # downward speed the hand sends the ball at, m/s

BALL_RADIUS = 0.12
BALL_MASS = 0.6
PADDLE_HALF = [0.15, 0.15, 0.02]

# ---------------------------------------------------------------------------
# Floor: the actual bounce surface. Passive restitution — no control here,
# this is what "returns" energy to the ball on the way back up.
# ---------------------------------------------------------------------------
plane = p.loadURDF("plane.urdf")
p.changeDynamics(plane, -1, restitution=FLOOR_E)

# Ball: HAND_HEIGHT is the *contact plane* -- where the ball's surface meets
# the paddle's underside -- not the center of either object. Spawn the ball
# one radius below it (touching, not overlapping), moving down at V_PUSH, as
# if the hand had just pushed it. This seeds the cycle with real energy
# instead of letting it fall from rest.
ball_col = p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_RADIUS)
ball_vis = p.createVisualShape(p.GEOM_SPHERE, radius=BALL_RADIUS, rgbaColor=[1, 0.4, 0, 1])
ball_start_z = HAND_HEIGHT - BALL_RADIUS - 0.001
ball = p.createMultiBody(BALL_MASS, ball_col, ball_vis, [0, 0, ball_start_z])
p.changeDynamics(ball, -1, restitution=FLOOR_E)  # combines with plane's restitution on floor contact
p.resetBaseVelocity(ball, linearVelocity=[0, 0, -V_PUSH])

# Hand / paddle: its center sits half its thickness ABOVE the contact plane,
# so its underside (not its center) is at HAND_HEIGHT.
paddle_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=PADDLE_HALF)
paddle_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=PADDLE_HALF, rgbaColor=[0.2, 0.2, 0.8, 1])
paddle_center_z = HAND_HEIGHT + PADDLE_HALF[2]
paddle = p.createMultiBody(0, paddle_col, paddle_vis, [0, 0, paddle_center_z])
p.changeDynamics(paddle, -1, restitution=0.0)  # this contact is fully hand-controlled, not passive

# ---------------------------------------------------------------------------
# Mirror-law hand controller: the hand sends the ball DOWN at a fixed speed
# regardless of how fast it arrives, by solving for the paddle velocity that
# produces the desired outgoing ball velocity.
#
#     v_ball_plus - v_p = -e * (v_ball_minus - v_p)
#     v_p = (v_ball_plus_desired + e * v_ball_minus) / (1 + e)
# ---------------------------------------------------------------------------

# Cosmetic-only: a brief visible dip of the paddle when it contacts the ball.
# Does not affect ball physics -- that's set directly via resetBaseVelocity.
PUSH_DEPTH = 0.05
PUSH_STEPS = 12
push_timer = 0

contact_cooldown = 0
paddle_z = paddle_center_z

apex_log = []       # (time, height) of each detected apex -- watch this for convergence
prev_vz = 0.0
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
    if contacts and contact_cooldown <= 0 and vz > 0:
        v_paddle_cmd = (-V_PUSH + HAND_E * vz) / (1 + HAND_E)
        v_ball_new = v_paddle_cmd * (1 + HAND_E) - HAND_E * vz  # should equal -V_PUSH

        p.resetBaseVelocity(ball, linearVelocity=[ball_vel[0], ball_vel[1], v_ball_new])
        contact_cooldown = 20
        push_timer = PUSH_STEPS
    else:
        contact_cooldown -= 1

    # Cosmetic paddle dip
    if push_timer > 0:
        paddle_z = paddle_center_z - PUSH_DEPTH * (push_timer / PUSH_STEPS)
        push_timer -= 1
    else:
        paddle_z = paddle_center_z
    p.resetBasePositionAndOrientation(paddle, [0, 0, paddle_z], [0, 0, 0, 1])

    time.sleep(DT)

p.disconnect()

print("Apex heights over time (hand height = {:.3f} m):".format(HAND_HEIGHT))
for t_apex, h in apex_log:
    print(f"  t={t_apex:6.2f}s  z={h:.3f}m")