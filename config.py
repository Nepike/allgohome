import numpy as np
from dataclasses import dataclass, field

GRID_W = 20
GRID_H = 20
CELL_SIZE = 40

DT = 0.5
MAX_LINEAR_SPEED = 1.4 # cells / sec
MAX_ANGULAR_SPEED = 3.0 # rad / sec
ANGULAR_SPEED_K = 1.3
WAYPOINT_THRESHOLD = 0.45 # cells
ANGLE_THRESHOLD = 0.25 # rad

MEAS_POS_STD = 5.4          # measurement noise in cells
MEAS_THETA_STD = 1.18        # rad
MEAS_DROP_PROB = 0.04        # occasional missed camera detection

KALMAN_Q = np.diag([0.2, 0.2, 0.03]) ** 2
KALMAN_R = np.diag([MEAS_POS_STD, MEAS_POS_STD, MEAS_THETA_STD]) ** 2

PF_POS_STD = 0.12
PF_THETA_STD = 0.06


@dataclass
class Robot:
    x: float # real pixels
    y: float # real pixels
    theta: float
    home_x: int # cell cords
    home_y: int # cell cords
    color: tuple


    true_path: list = field(default_factory=list)  # real pixels
    astar_path: list = field(default_factory=list) # cells cords
    measurements: list = field(default_factory=list)  # real pixels

    ekf_path: list = field(default_factory=list)
    ukf_path: list = field(default_factory=list)
    pf_path: list = field(default_factory=list)
