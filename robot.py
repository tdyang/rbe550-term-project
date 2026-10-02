"""Robot: chassis + fixed-length arm (variable angle) + paddle."""

import math

import numpy as np
import pybullet as p

from config import ARM_LENGTH, BASE_XY_HALF, PADDLE_CENTER_Z, PADDLE_HALF


class Robot:
    """Kinematic base + paddle on a fixed-length arm whose *angle* (phi) is
    chosen independently of the ball's push direction each step -- unlike a
    rigid heading+90-degree offset, this lets the chassis fall in behind the
    ball instead of beside it when the side is too tight. Chassis and
    paddle share phi as their orientation, since the arm rigidly connects
    them into one assembly that turns together. The arm is rendered as a
    debug line (not a rotated box) since it trivially handles any phi
    without solving for box orientation."""

    def __init__(self, arm_length=ARM_LENGTH, base_xy_half=BASE_XY_HALF):
        self.arm_length = arm_length
        self.base_xy_half = base_xy_half
        base_half = [base_xy_half, base_xy_half, PADDLE_CENTER_Z / 2]
        self.base_z = base_half[2]

        base_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=base_half, rgbaColor=[0.2, 0.7, 0.3, 1])
        self.base_id = p.createMultiBody(0, -1, base_vis, [0, arm_length, self.base_z])

        paddle_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=PADDLE_HALF)
        paddle_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=PADDLE_HALF, rgbaColor=[0.2, 0.2, 0.8, 1])
        self.paddle_id = p.createMultiBody(0, paddle_col, paddle_vis, [0, 0, PADDLE_CENTER_Z])
        p.changeDynamics(self.paddle_id, -1, restitution=0.0)

        self.arm_line_id = -1
        self.base_xy = np.array([0.0, arm_length])

    def set_pose(self, ball_xy, phi, paddle_z=PADDLE_CENTER_Z):
        """ball_xy: the paddle's position (== the ball's own, since they
        must coincide). phi: world-frame angle from the paddle to the
        chassis. Both the chassis and the paddle are oriented by phi --
        they're one rigid assembly connected by the arm, so they turn
        together, rather than the paddle spinning to face the ball's push
        direction independently of which way the arm/chassis are facing."""
        paddle_xy = np.array(ball_xy)
        unit = np.array([math.cos(phi), math.sin(phi)])
        base_xy = paddle_xy + self.arm_length * unit
        self.base_xy = base_xy

        quat = p.getQuaternionFromEuler([0, 0, phi])
        p.resetBasePositionAndOrientation(self.base_id, [*base_xy, self.base_z], quat)
        p.resetBasePositionAndOrientation(self.paddle_id, [*paddle_xy, paddle_z], quat)

        base_face = base_xy - self.base_xy_half * unit   # line starts at the chassis edge, not its center
        self.arm_line_id = p.addUserDebugLine(
            [*base_face, PADDLE_CENTER_Z], [*paddle_xy, PADDLE_CENTER_Z],
            lineColorRGB=[0.2, 0.7, 0.3], lineWidth=6, replaceItemUniqueId=self.arm_line_id)
        return base_xy

    def touches(self, other_id):
        return bool(p.getContactPoints(self.paddle_id, other_id))
