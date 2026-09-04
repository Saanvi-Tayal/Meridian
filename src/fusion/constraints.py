"""Vehicle Kinematic Constraints (NHC, ZUPT, AI-Speed)

Implements kinematic observation models and measurement Jacobians for ground vehicles:
1. Non-Holonomic Constraints (NHC): Lateral velocity v_x^B = 0 and vertical velocity v_z^B = 0.
2. Zero Velocity Updates (ZUPT): 3D velocity v^N = 0 during vehicle standstill.
3. Forward Speed Constraint: Longitudinal velocity v_y^B = v_AI.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


class NonHolonomicConstraint:
    """
    Ground vehicle Non-Holonomic Constraint (NHC).
    Assumes wheels do not slide sideways (v_x^B = 0) and cannot fly off road (v_z^B = 0).
    v^B = R(q)^T * v^N
    """

    def __init__(self, sigma_lateral: float = 0.15, sigma_vertical: float = 0.15):
        # Measurement noise covariance (2x2)
        self.R = np.diag([sigma_lateral**2, sigma_vertical**2])

    def get_measurement_model(self, R_nb: np.ndarray, v_n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        R_nb: 3x3 rotation matrix from Body to Navigation frame (R_nb @ v_b = v_n)
              Therefore v_b = R_nb.T @ v_n.
        v_n:  Current estimated velocity in navigation frame (3,)
        returns: (y, H, R)
          y: Innovation vector (2,) = z - h(x) = [0 - v_x^B, 0 - v_z^B]
          H: Jacobian matrix (2, 15) with respect to error state [dp, dv, dtheta, dba, dbg]
          R: Noise covariance (2, 2)
        """
        R_bn = R_nb.T # From Nav to Body
        v_b = R_bn @ v_n

        # True pseudo-measurement is [0, 0]
        # h(x) = [v_b[0], v_b[2]]
        # Innovation y = z - h(x) = [0 - v_b[0], 0 - v_b[2]]
        y = np.array([-v_b[0], -v_b[2]])

        # Jacobian w.r.t [dp, dv, dtheta, dba, dbg] (15 states):
        # dv_b / dv_n = R_bn
        # dv_b / dtheta = [R_bn * v_n]_x = [v_b]_x
        H = np.zeros((2, 15))
        # w.r.t velocity (columns 3, 4, 5)
        H[0, 3:6] = R_bn[0, :]
        H[1, 3:6] = R_bn[2, :]

        # w.r.t attitude error (columns 6, 7, 8)
        # delta v_b = R_bn * (- [delta_theta]_x * v_n) = [v_b]_x * delta_theta
        v_b_cross = np.array([
            [0.0, -v_b[2], v_b[1]],
            [v_b[2], 0.0, -v_b[0]],
            [-v_b[1], v_b[0], 0.0]
        ])
        H[0, 6:9] = v_b_cross[0, :]
        H[1, 6:9] = v_b_cross[2, :]

        return y, H, self.R


class ZeroVelocityUpdate:
    """
    Zero Velocity Update (ZUPT).
    Applied when the vehicle is stopped (traffic lights, stops).
    Enforces v^N = [0, 0, 0]^T.
    """

    def __init__(self, sigma_zupt: float = 0.02):
        self.R = np.eye(3) * (sigma_zupt**2)

    def get_measurement_model(self, v_n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        returns: (y, H, R)
          y: Innovation (3,) = [0, 0, 0] - v_n
          H: Jacobian (3, 15)
          R: Noise covariance (3, 3)
        """
        y = -v_n
        H = np.zeros((3, 15))
        H[:, 3:6] = np.eye(3) # Direct observation of velocity error delta_v

        return y, H, self.R


class AISpeedConstraint:
    """
    Longitudinal Forward Speed Constraint from AI Model (SpeedNet).
    v_y^B = R_bn[1, :] @ v_n = v_AI
    """

    def __init__(self, sigma_speed: float = 1.2):
        self.R = np.array([[sigma_speed**2]])

    def get_measurement_model(self, R_nb: np.ndarray, v_n: np.ndarray, v_ai: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        returns: (y, H, R)
          y: Innovation (1,) = v_ai - v_y^B
          H: Jacobian (1, 15)
          R: Noise covariance (1, 1)
        """
        R_bn = R_nb.T
        v_b = R_bn @ v_n
        y = np.array([v_ai - v_b[1]])

        H = np.zeros((1, 15))
        # w.r.t velocity
        H[0, 3:6] = R_bn[1, :]

        # w.r.t attitude error
        v_b_cross = np.array([
            [0.0, -v_b[2], v_b[1]],
            [v_b[2], 0.0, -v_b[0]],
            [-v_b[1], v_b[0], 0.0]
        ])
        H[0, 6:9] = v_b_cross[1, :]

        return y, H, self.R
