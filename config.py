"""Shared physical/world constants used by more than one module. Constants
only one module cares about (hybrid-A* lattice resolution, planning safety
margins, etc.) stay local to that module instead of living here."""

import math

DT = 1.0 / 240.0

# --- Tunable parameters -----------------------------------------------------
FLOOR_E = 0.85
HAND_HEIGHT = 0.6
HAND_E = 0.8
V_PUSH = 3.0
LANE_VX = 0.25

BALL_RADIUS = 0.12
BALL_MASS = 0.6
PADDLE_HALF = [0.15, 0.15, 0.02]
BASE_XY_HALF = 0.2

G = 9.81
CONTACT_Z = HAND_HEIGHT - BALL_RADIUS
FLOOR_Z = BALL_RADIUS
PADDLE_CENTER_Z = HAND_HEIGHT + PADDLE_HALF[2]

COURSE_LENGTH = 6.0
COURSE_HALF_WIDTH = 1.5   # obstacles/path may range y in [-COURSE_HALF_WIDTH, COURSE_HALF_WIDTH]
ARM_LENGTH = 0.5          # fixed distance the robot escorts the ball at
N_OBSTACLES_RANGE = (6, 9)
ROBOT_RADIUS = BASE_XY_HALF * math.sqrt(2) + 0.03   # chassis circumscribed circle + safety margin
BALL_CLEARANCE = BALL_RADIUS + 0.05
