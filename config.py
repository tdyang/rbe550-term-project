"""Shared physical/world constants used by more than one module. Constants
only one module cares about (hybrid-A* lattice resolution, planning safety
margins, etc.) stay local to that module instead of living here."""

import math

DT = 1.0 / 240.0

# --- Tunable parameters -----------------------------------------------------
FLOOR_E = 0.85
HAND_HEIGHT = 0.6
HAND_E = 0.8
# Ball speed off the paddle (the mirror law's constant). It sets the
# dribble's pace: 2.2 m/s gives a 0.30s bounce cycle (3.3 per second; 3.0
# gave 0.23s). Much lower and the floor rebound barely reaches the paddle
# -- at 2.0 the ball would peak only 4.7cm above it, at 1.65 not at all.
V_PUSH = 2.2
LANE_VX = 0.25   # nominal travel speed; the actual cruise speed adapts to clearance (below)

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

# Dynamic obstacles (Objective 4 stretch goal): a fraction of obstacles
# oscillate back and forth along the course's y-axis on a known sinusoidal
# schedule (fixed amplitude/period/phase per obstacle, drawn at course-
# generation time), rather than moving unpredictably -- deterministic
# motion is what lets the planner treat time as a 4th search dimension and
# prove the whole run collision-free in advance, the same guarantee the
# static case already has, instead of falling back to a live, reactive
# layer with no such guarantee.
DYNAMIC_OBSTACLE_FRACTION = 0.5
# Peak obstacle speed is 2*pi*amplitude/period: these ranges cap it at
# ~0.94 m/s (typically ~0.5), still 2-4x the robot's own LANE_VX. The
# original (0.3-0.8m, 2-5s) reached ~2.5 m/s, at which even a 0.1s gap
# between planned and real timing is a quarter-meter position error -- no
# realistic time margin (planning.TIME_MARGIN) could be reserved for that
# and still leave a solvable course.
DYNAMIC_AMPLITUDE_RANGE = (0.3, 0.6)   # meters, peak offset from the obstacle's rest position
DYNAMIC_PERIOD_RANGE = (4.0, 8.0)      # seconds, full back-and-forth cycle

# Fastest the ball may be pushed horizontally at a contact. LANE_VX is the
# nominal pace the planner schedules against; the live controller pushes
# faster or slower than that to stay on the plan's clock (simulation.py's
# plan_next_contact), and the planner refuses any arm-angle transition
# that would need the ball to move faster than this to keep up
# (planning.plan_phi_sequence) -- so every planned schedule is one the
# controller can actually track.
# Set with headroom over the fastest cruise (CRUISE_SPEED_RANGE[1]) for
# the arm to swing the ball sideways: it was 2*LANE_VX (0.5 m/s) when
# travel topped out at 0.25, and left at 0.5 once cruise reached 0.45 it
# left the arm so little room that the offline planner's arm search failed
# on 23 of 60 courses (from 8). A 0.7 m/s push needs well under the
# paddle's tilt limit.
BALL_SPEED_MAX = 0.7

# Travel speed adapts to how open the way is: cruise at CRUISE_SPEED_RANGE[0]
# where the robot body has CLEARANCE_RANGE[0] or less of free space to the
# nearest static obstacle, CRUISE_SPEED_RANGE[1] beyond CLEARANCE_RANGE[1],
# and in proportion between (planning.cruise_speed). Both planners use it:
# the offline one by taking longer over each lattice step in tight spots,
# the live one by setting each contact's cruise speed from the route ahead.
# On 30 roaming courses, ramping over 0-0.5m gave the fewest collisions (21
# in 7 runs, vs. 25 in 12 at a fixed 0.25 m/s and 31 in 10 over 0.15-0.8m,
# where the dense courses kept the robot near its slowest). Over 90
# courses, raising the open-ground speed from 0.35 to 0.45 m/s together
# with live_planning's chase speed-up finished as many (84 vs. 83) about
# 10% faster, at the cost of more collisions (72 vs. 53).
CRUISE_SPEED_RANGE = (0.15, 0.45)   # m/s
CLEARANCE_RANGE = (0.0, 0.5)        # m

# Mobile-base limits. The chassis's yaw is the arm angle phi (chassis and
# arm turn together as one rigid assembly), so the turn-rate limit caps how
# fast phi can change; the speed limit caps how far the chassis moves
# between contacts. Both planners respect them.
BASE_SPEED_MAX = 0.5                      # m/s
BASE_TURN_RATE_MAX = math.radians(90)     # rad/s

# Paddle actuator limits for the energy-consistent contact model
# (physics.paddle_impact): the paddle face tilts to redirect the ball, and
# its normal speed at impact is set by the restitution law.
PADDLE_TILT_MAX = math.radians(30)        # face tilt from horizontal
PADDLE_SPEED_MAX = 3.0                    # m/s along the face normal

# Roaming obstacles (live planner, live_planning.py): a moving obstacle
# heads off in a random direction at constant speed, bouncing off the
# course's edges, the static obstacles, and each other. The motion is
# unknown to the live planner, which only senses where each obstacle is
# now. Roaming obstacles stay within ROAM_X_RANGE so they never sweep
# across the start or past the finish.
# Each roaming obstacle gets its own base speed from ROAM_SPEED_RANGE, from
# slow wanderers to fast movers quicker than the robot's nominal LANE_VX
# (though under its 0.45 m/s evasive speed). This range used to be capped at
# the robot's pace (0.10-0.25) so nothing could run it down from behind;
# widening it trades some of that safety for visibly different obstacles.
ROAM_SPEED_RANGE = (0.05, 0.40)   # m/s, per-obstacle base speed
ROAM_X_RANGE = (1.0, COURSE_LENGTH - 0.3)
# Random turns: each roaming obstacle also changes course at random moments
# (a Poisson process at ROAM_TURN_RATE), swinging its heading by up to
# +-ROAM_TURN_MAX and changing speed by up to +-ROAM_SPEED_JITTER of its
# own base speed, so a fast obstacle stays fast and a slow one slow --
# motion the live planner's bounce-aware prediction can't foresee.
ROAM_TURN_RATE = 0.4                 # turns per second (mean 2.5s apart)
ROAM_TURN_MAX = math.radians(120)
ROAM_SPEED_JITTER = 0.3
