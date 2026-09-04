"""In-Vehicle Alignment & Dynamic Calibration Engine

Automatically determines the rotation matrix R_s2v between the smartphone's arbitrary
mounting frame {S} and the vehicle's kinematic body frame {V} (X: Right, Y: Forward, Z: Up).
Supports initial calibration and continuous online mount shift detection.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class CalibrationResult:
    """Stores the calibration state and rotation matrices."""
    is_calibrated: bool
    R_s2v: np.ndarray             # 3x3 rotation matrix: v_vehicle = R_s2v @ v_sensor
    pitch_deg: float              # Estimated pitch relative to horizontal
    roll_deg: float               # Estimated roll relative to horizontal
    yaw_misalignment_deg: float   # Heading angle offset between phone and vehicle forward
    gravity_norm: float           # Measured gravity magnitude (m/s^2)


class InVehicleAligner:
    """
    Two-stage in-vehicle alignment engine:
    1. Leveling (Pitch & Roll): Computes vertical axis from gravity vector.
    2. Heading Alignment (Yaw): Identifies vehicle forward direction from longitudinal acceleration dynamics.
    """

    def __init__(
        self,
        stationary_acc_var_thresh: float = 0.08,
        stationary_gyro_var_thresh: float = 0.005,
        min_accel_dynamics_thresh: float = 0.5, # m/s^2
        mount_shift_angle_thresh_deg: float = 12.0,
    ):
        self.acc_var_thresh = stationary_acc_var_thresh
        self.gyro_var_thresh = stationary_gyro_var_thresh
        self.min_accel_thresh = min_accel_dynamics_thresh
        self.mount_shift_thresh_rad = np.radians(mount_shift_angle_thresh_deg)

        self.R_s2v: Optional[np.ndarray] = None
        self.u_z_sensor: Optional[np.ndarray] = None
        self.u_y_sensor: Optional[np.ndarray] = None
        self.is_calibrated: bool = False

    def estimate_vertical_axis(
        self,
        acc_window: np.ndarray,
        gyro_window: Optional[np.ndarray] = None,
        gravity_window: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Calculates the vertical unit vector (Up) in the smartphone sensor frame.
        If Android synthetic gravity is available, uses it directly; otherwise averages accelerometer.
        """
        if gravity_window is not None and len(gravity_window) > 0:
            mean_g = np.mean(gravity_window, axis=0)
            g_norm = np.linalg.norm(mean_g)
            if g_norm > 1.0:
                # Sensor measures gravity vector pointing downwards; Up is opposite or along depending on convention.
                # Standard convention: Vehicle Z is UP, sensor accelerometer at rest measures +9.81 m/s^2 upwards reaction force.
                u_z = mean_g / g_norm
                return u_z, float(g_norm)

        # Fallback: estimate during stationary or near-constant velocity
        mean_acc = np.mean(acc_window, axis=0)
        g_norm = np.linalg.norm(mean_acc)
        u_z = mean_acc / (g_norm + 1e-8)
        return u_z, float(g_norm)

    def estimate_forward_axis(
        self,
        acc: np.ndarray,
        u_z: np.ndarray,
        speed_ref: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Identifies vehicle forward unit vector (Longitudinal) in the smartphone frame.
        Projects acceleration onto the horizontal plane and uses PCA / variance maximization.
        Resolves forward/backward ambiguity using speed correlation.
        """
        # 1. Project acceleration onto horizontal plane: a_horiz = a - (a . u_z) * u_z
        z_component = np.sum(acc * u_z, axis=1, keepdims=True)
        acc_horiz = acc - z_component * u_z

        # 2. Filter for significant vehicle acceleration / braking events
        acc_horiz_mag = np.linalg.norm(acc_horiz, axis=1)
        dynamic_mask = acc_horiz_mag > self.min_accel_thresh

        if np.sum(dynamic_mask) < 20:
            # Not enough dynamics, use horizontal vector with highest overall variance
            cov = np.cov(acc_horiz.T)
        else:
            cov = np.cov(acc_horiz[dynamic_mask].T)

        eigvals, eigvecs = np.linalg.eigh(cov)
        # Principal component with largest variance in horizontal plane
        u_long = eigvecs[:, -1]
        # Ensure it is strictly orthogonal to u_z
        u_long = u_long - np.dot(u_long, u_z) * u_z
        u_long = u_long / (np.linalg.norm(u_long) + 1e-8)

        # 3. Resolve Sign (+Forward vs -Backward)
        # In vehicle braking/acceleration, forward acceleration a_y corresponds to d(v)/dt.
        if speed_ref is not None and len(speed_ref) == len(acc):
            dv = np.gradient(speed_ref)
            proj_acc = np.dot(acc_horiz, u_long)
            if np.std(dv) > 1e-4 and np.std(proj_acc) > 1e-4:
                corr = np.corrcoef(dv, proj_acc)[0, 1]
                if not np.isnan(corr) and corr < 0:
                    u_long = -u_long
        else:
            # Fallback: by typical smartphone portrait dashboard mount, +Y or -Z points forward
            pass

        return u_long

    def calibrate(
        self,
        acc: np.ndarray,
        gyro: Optional[np.ndarray] = None,
        gravity: Optional[np.ndarray] = None,
        speed_ref: Optional[np.ndarray] = None,
    ) -> CalibrationResult:
        """
        Runs full in-vehicle alignment calibration given a calibration sequence.
        Returns the rotation matrix R_s2v mapping: v_vehicle = R_s2v @ v_sensor.
        """
        # Step 1: Up axis (Z_v) in sensor frame
        u_z, g_norm = self.estimate_vertical_axis(acc, gyro, gravity)
        self.u_z_sensor = u_z

        # Step 2: Forward axis (Y_v) in sensor frame
        u_y = self.estimate_forward_axis(acc, u_z, speed_ref)
        self.u_y_sensor = u_y

        # Step 3: Lateral axis (X_v = Y_v x Z_v) to complete right-handed coordinate frame
        u_x = np.cross(u_y, u_z)
        u_x = u_x / (np.linalg.norm(u_x) + 1e-8)

        # Re-orthogonalize Z_v = X_v x Y_v to guarantee perfect orthonormality
        u_z = np.cross(u_x, u_y)
        u_z = u_z / (np.linalg.norm(u_z) + 1e-8)
        self.u_z_sensor = u_z

        # Construct rotation matrix: rows are vehicle basis vectors expressed in sensor frame
        # v_vehicle = [dot(v, u_x); dot(v, u_y); dot(v, u_z)] = R_s2v @ v
        self.R_s2v = np.vstack([u_x, u_y, u_z])
        self.is_calibrated = True

        # Extract Euler angles (Roll, Pitch, Yaw)
        pitch = np.degrees(np.arcsin(-self.R_s2v[2, 1]))
        roll = np.degrees(np.arctan2(self.R_s2v[2, 0], self.R_s2v[2, 2]))
        yaw = np.degrees(np.arctan2(self.R_s2v[0, 1], self.R_s2v[1, 1]))

        return CalibrationResult(
            is_calibrated=True,
            R_s2v=self.R_s2v,
            pitch_deg=float(pitch),
            roll_deg=float(roll),
            yaw_misalignment_deg=float(yaw),
            gravity_norm=g_norm
        )

    def transform_vector(self, sensor_vector: np.ndarray) -> np.ndarray:
        """
        Rotates a vector or batch of vectors from sensor frame {S} to vehicle frame {V}.
        sensor_vector: Shape (3,) or (N, 3)
        returns: Transformed vector(s) in vehicle frame (X: Right, Y: Forward, Z: Up).
        """
        if not self.is_calibrated or self.R_s2v is None:
            raise RuntimeError("InVehicleAligner is not calibrated yet. Call calibrate() first.")
        
        if sensor_vector.ndim == 1:
            return self.R_s2v @ sensor_vector
        return (self.R_s2v @ sensor_vector.T).T

    def detect_mount_shift(self, current_acc_window: np.ndarray) -> bool:
        """
        Checks if the phone has shifted or fallen on its mount by monitoring gravity vector angular deviation.
        """
        if not self.is_calibrated or self.u_z_sensor is None:
            return False

        new_u_z, _ = self.estimate_vertical_axis(current_acc_window)
        cos_angle = np.clip(np.dot(new_u_z, self.u_z_sensor), -1.0, 1.0)
        angle_dev = np.arccos(cos_angle)

        return bool(angle_dev > self.mount_shift_thresh_rad)
