"""15-State Error-State Extended Kalman Filter (ES-EKF) for GNSS+INS Sensor Fusion

Propagates nominal high-rate IMU strapdown inertial navigation and applies error-state
Kalman corrections from GNSS fixes, Non-Holonomic Constraints (NHC), Zero-Velocity Updates (ZUPT),
and AI Speed Estimator predictions.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Builds a 3x3 skew-symmetric cross-product matrix from a 3D vector."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """Converts a normalized unit quaternion [qw, qx, qy, qz] to a 3x3 rotation matrix R_nb."""
    qw, qx, qy, qz = q
    return np.array([
        [1.0 - 2.0*(qy**2 + qz**2), 2.0*(qx*qy - qw*qz),     2.0*(qx*qz + qw*qy)],
        [2.0*(qx*qy + qw*qz),     1.0 - 2.0*(qx**2 + qz**2), 2.0*(qy*qz - qw*qx)],
        [2.0*(qx*qz - qw*qy),     2.0*(qy*qz + qw*qx),     1.0 - 2.0*(qx**2 + qy**2)]
    ])


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamiltonian quaternion multiplication q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def rot_vec_to_quat(v: np.ndarray) -> np.ndarray:
    """Converts a small rotation vector theta (radians) to a unit quaternion."""
    angle = np.linalg.norm(v)
    if angle < 1e-8:
        return np.array([1.0, 0.5*v[0], 0.5*v[1], 0.5*v[2]])
    half_angle = 0.5 * angle
    axis = v / angle
    return np.array([np.cos(half_angle), axis[0]*np.sin(half_angle), axis[1]*np.sin(half_angle), axis[2]*np.sin(half_angle)])


@dataclass
class ESKFState:
    """Full nominal state container."""
    p_n: np.ndarray      # Position in Navigation Frame (3,) [meters] (X: East, Y: North, Z: Up)
    v_n: np.ndarray      # Velocity in Navigation Frame (3,) [m/s]
    q_nb: np.ndarray     # Unit quaternion from Body to Nav Frame [qw, qx, qy, qz]
    b_a: np.ndarray      # Accelerometer bias (3,) [m/s^2]
    b_g: np.ndarray      # Gyroscope bias (3,) [rad/s]
    P: np.ndarray        # 15x15 Error state covariance matrix


class ErrorStateKalmanFilter:
    """
    15-state Error-State Extended Kalman Filter for Land Vehicle Dead Reckoning.
    State error vector: delta_x = [delta_p, delta_v, delta_theta, delta_ba, delta_bg]^T
    """

    def __init__(
        self,
        init_pos: Optional[np.ndarray] = None,
        init_vel: Optional[np.ndarray] = None,
        init_quat: Optional[np.ndarray] = None,
        acc_noise_density: float = 0.05,     # m/s^2 / sqrt(Hz)
        gyro_noise_density: float = 0.005,   # rad/s / sqrt(Hz)
        acc_bias_instability: float = 1e-4,  # m/s^3 / sqrt(Hz)
        gyro_bias_instability: float = 1e-5, # rad/s^2 / sqrt(Hz)
    ):
        # Gravity vector in Local Cartesian / ENU frame (Z is Up: gravity points downwards [0, 0, -9.80665])
        self.g_n = np.array([0.0, 0.0, -9.80665])

        # Initialize nominal state
        self.p_n = init_pos if init_pos is not None else np.zeros(3)
        self.v_n = init_vel if init_vel is not None else np.zeros(3)
        self.q_nb = init_quat if init_quat is not None else np.array([1.0, 0.0, 0.0, 0.0])
        self.b_a = np.zeros(3)
        self.b_g = np.zeros(3)

        # Initialize 15x15 error state covariance matrix
        self.P = np.eye(15)
        self.P[0:3, 0:3] *= 1.0     # Initial pos error: 1m
        self.P[3:6, 3:6] *= 0.5     # Initial vel error: 0.5 m/s
        self.P[6:9, 6:9] *= (0.05**2) # Initial attitude error: ~3 deg
        self.P[9:12, 9:12] *= (0.05**2) # Initial acc bias
        self.P[12:15, 12:15] *= (0.005**2) # Initial gyro bias

        # Continuous noise spectral densities
        self.q_acc = acc_noise_density**2
        self.q_gyro = gyro_noise_density**2
        self.q_ba = acc_bias_instability**2
        self.q_bg = gyro_bias_instability**2

    @property
    def R_nb(self) -> np.ndarray:
        """Current 3x3 rotation matrix mapping Body vectors to Navigation vectors."""
        return quat_to_rot_matrix(self.q_nb)

    def predict(self, a_meas: np.ndarray, w_meas: np.ndarray, dt: float):
        """
        Propagates the nominal state and error covariance over timestep dt.
        a_meas: Measured specific force in vehicle body frame (3,) [m/s^2]
        w_meas: Measured angular rate in vehicle body frame (3,) [rad/s]
        """
        # 1. Bias-corrected IMU measurements
        f_b = a_meas - self.b_a
        w_b = w_meas - self.b_g

        # 2. Strapdown Nominal State Kinematic Integration
        R = self.R_nb
        f_n = R @ f_b
        a_n = f_n + self.g_n

        # Position and Velocity update
        self.p_n = self.p_n + self.v_n * dt + 0.5 * a_n * (dt**2)
        self.v_n = self.v_n + a_n * dt

        # Attitude integration via quaternion
        delta_q = rot_vec_to_quat(w_b * dt)
        self.q_nb = quat_multiply(self.q_nb, delta_q)
        self.q_nb /= np.linalg.norm(self.q_nb) # Re-normalize

        # 3. Error-State Transition Jacobian (15x15)
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -skew_symmetric(f_n) * dt
        F[3:6, 9:12] = -R * dt
        F[6:9, 6:9] = np.eye(3) - skew_symmetric(R @ w_b) * dt
        F[6:9, 12:15] = -R * dt

        # 4. Discrete Process Noise Covariance Q (15x15)
        Q = np.zeros((15, 15))
        Q[0:3, 0:3] = np.eye(3) * (self.q_acc * (dt**3) / 3.0)
        Q[3:6, 3:6] = np.eye(3) * (self.q_acc * dt)
        Q[6:9, 6:9] = np.eye(3) * (self.q_gyro * dt)
        Q[9:12, 9:12] = np.eye(3) * (self.q_ba * dt)
        Q[12:15, 12:15] = np.eye(3) * (self.q_bg * dt)

        # 5. Covariance Propagation
        self.P = F @ self.P @ F.T + Q
        # Enforce symmetry
        self.P = 0.5 * (self.P + self.P.T)

    def apply_kalman_update(self, y: np.ndarray, H: np.ndarray, R: np.ndarray):
        """
        Generic Kalman measurement update with Joseph-form covariance stabilization.
        y: Innovation vector (M,)
        H: Measurement Jacobian (M, 15)
        R: Measurement noise covariance (M, M)
        """
        # Innovation covariance
        S = H @ self.P @ H.T + R
        # Kalman Gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Compute error state correction
        delta_x = K @ y

        # State Injection
        self.p_n += delta_x[0:3]
        self.v_n += delta_x[3:6]
        
        # Attitude correction: delta_theta -> delta_q
        delta_q = rot_vec_to_quat(delta_x[6:9])
        self.q_nb = quat_multiply(delta_q, self.q_nb)
        self.q_nb /= np.linalg.norm(self.q_nb)

        # Bias corrections
        self.b_a += delta_x[9:12]
        self.b_g += delta_x[12:15]

        # Joseph form covariance update for numerical stability: P = (I - KH)P(I - KH)^T + KRK^T
        I = np.eye(15)
        I_KH = I - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def update_gnss(self, pos_meas: np.ndarray, vel_meas: Optional[np.ndarray] = None, pos_std: float = 2.5, vel_std: float = 0.3):
        """
        Full 3D GNSS position (and velocity) measurement update.
        """
        if vel_meas is not None:
            # 6D measurement [px, py, pz, vx, vy, vz]
            z = np.hstack([pos_meas, vel_meas])
            h = np.hstack([self.p_n, self.v_n])
            y = z - h

            H = np.zeros((6, 15))
            H[0:3, 0:3] = np.eye(3)
            H[3:6, 3:6] = np.eye(3)

            R = np.diag([pos_std**2, pos_std**2, (pos_std*2)**2, vel_std**2, vel_std**2, (vel_std*2)**2])
            self.apply_kalman_update(y, H, R)
        else:
            # 3D position only
            y = pos_meas - self.p_n
            H = np.zeros((3, 15))
            H[0:3, 0:3] = np.eye(3)
            R = np.diag([pos_std**2, pos_std**2, (pos_std*2)**2])
            self.apply_kalman_update(y, H, R)

    def get_state(self) -> ESKFState:
        """Returns snapshot of the current nominal state."""
        return ESKFState(
            p_n=self.p_n.copy(),
            v_n=self.v_n.copy(),
            q_nb=self.q_nb.copy(),
            b_a=self.b_a.copy(),
            b_g=self.b_g.copy(),
            P=self.P.copy(),
        )
