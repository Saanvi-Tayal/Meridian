"""GNSS Reacquisition Adaptive Calibration & Error-Feedback Engine

When GNSS fix is restored after an outage (e.g., exiting a tunnel or underpass):
1. Compares the dead-reckoning predicted exit state with the true GNSS fix.
2. Computes the along-track (speed scale) and cross-track (heading / gyro bias) errors.
3. Dynamically updates sensor calibration parameters (gyro bias, AI speed scale) so subsequent
   predictions become progressively more accurate (closed-loop learning).
4. Provides smooth trajectory stitching (anti-teleport) to prevent jarring jumps on the UI map.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np


def wrap_angle_rad(angle: float) -> float:
    """Wraps angle to [-pi, pi]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class OutageSummary:
    """Detailed error diagnostics computed upon GNSS reacquisition."""
    entry_time_s: float
    exit_time_s: float
    duration_s: float
    distance_traveled_m: float
    
    # State vectors: [East, North]
    predicted_exit_pos: np.ndarray
    predicted_exit_speed: float
    predicted_exit_heading_rad: float
    
    gnss_exit_pos: np.ndarray
    gnss_exit_speed: float
    gnss_exit_heading_rad: float
    
    # Discrepancy metrics
    pos_error_vector: np.ndarray  # p_gnss - p_pred
    total_pos_error_m: float
    along_track_error_m: float    # Speed / scale factor error
    cross_track_error_m: float    # Lateral / heading drift error
    heading_error_rad: float
    speed_error_ms: float
    drift_percent: float


@dataclass
class CalibrationUpdate:
    """Adaptive calibration parameter updates derived from outage error feedback."""
    delta_gyro_bias_rad_s: float
    new_gyro_bias_rad_s: float
    speed_scale_factor: float
    new_speed_scale: float


class GNSSReacquisitionManager:
    """
    Manages GNSS outages, reacquisition detection, adaptive parameter calibration,
    and anti-teleport UI trajectory stitching.
    """

    def __init__(
        self,
        gyro_learning_rate: float = 0.5,
        speed_learning_rate: float = 0.5,
        smoothing_decay: float = 2.0,
        min_outage_duration_s: float = 2.0,
    ):
        """
        Args:
            gyro_learning_rate: Proportion of heading error rate assigned to gyro bias update (0 < lr <= 1).
            speed_learning_rate: Proportion of relative along-track error assigned to speed scale update.
            smoothing_decay: Exponential decay rate lambda for UI visual marker smoothing (1/s).
            min_outage_duration_s: Minimum duration required to trigger parameter adaptation.
        """
        self.gyro_lr = gyro_learning_rate
        self.speed_lr = speed_learning_rate
        self.smoothing_decay = smoothing_decay
        self.min_outage_duration = min_outage_duration_s

        # Active calibration parameters
        self.active_gyro_bias_rad_s: float = 0.0
        self.active_speed_scale: float = 1.0

        # Outage tracking state
        self.is_in_outage: bool = False
        self.entry_time_s: float = 0.0
        self.entry_pos: Optional[np.ndarray] = None
        self.entry_heading_rad: float = 0.0

        # Reacquisition smoothing state
        self.reacquisition_time_s: float = -1e9
        self.last_pos_error_vec: np.ndarray = np.zeros(2)

        # History of past outages
        self.outage_history: List[OutageSummary] = []

    def notify_gnss_lost(
        self,
        time_s: float,
        p_entry: np.ndarray,
        heading_entry_rad: float
    ):
        """Called when GNSS signal drops (entering tunnel or canyon)."""
        self.is_in_outage = True
        self.entry_time_s = time_s
        self.entry_pos = np.array(p_entry, dtype=float)
        self.entry_heading_rad = heading_entry_rad

    def notify_gnss_restored(
        self,
        exit_time_s: float,
        p_pred_exit: np.ndarray,
        v_pred_exit: float,
        heading_pred_exit_rad: float,
        distance_pred_m: float,
        p_gnss_exit: np.ndarray,
        v_gnss_exit: float,
        heading_gnss_exit_rad: float
    ) -> Tuple[OutageSummary, CalibrationUpdate]:
        """
        Called when GNSS fix is restored. Compares prediction against GNSS fix,
        derives closed-loop calibration updates, and prepares UI smoothing.
        """
        self.is_in_outage = False
        duration_s = max(1e-3, exit_time_s - self.entry_time_s)

        p_pred = np.array(p_pred_exit, dtype=float)
        p_gnss = np.array(p_gnss_exit, dtype=float)

        # 1. Error Vector: from prediction to ground truth
        pos_err_vec = p_gnss - p_pred
        total_pos_err = float(np.linalg.norm(pos_err_vec))

        # 2. Project onto along-track and cross-track directions based on predicted exit heading
        # Heading psi: 0 is North (Y), pi/2 is East (X)
        u_along = np.array([np.sin(heading_pred_exit_rad), np.cos(heading_pred_exit_rad)])
        u_cross = np.array([np.cos(heading_pred_exit_rad), -np.sin(heading_pred_exit_rad)])

        along_track_err = float(np.dot(pos_err_vec, u_along))
        cross_track_err = float(np.dot(pos_err_vec, u_cross))

        heading_err = wrap_angle_rad(heading_gnss_exit_rad - heading_pred_exit_rad)
        speed_err = v_gnss_exit - v_pred_exit
        drift_pct = (total_pos_err / max(1.0, distance_pred_m)) * 100.0

        summary = OutageSummary(
            entry_time_s=self.entry_time_s,
            exit_time_s=exit_time_s,
            duration_s=duration_s,
            distance_traveled_m=distance_pred_m,
            predicted_exit_pos=p_pred,
            predicted_exit_speed=v_pred_exit,
            predicted_exit_heading_rad=heading_pred_exit_rad,
            gnss_exit_pos=p_gnss,
            gnss_exit_speed=v_gnss_exit,
            gnss_exit_heading_rad=heading_gnss_exit_rad,
            pos_error_vector=pos_err_vec,
            total_pos_error_m=total_pos_err,
            along_track_error_m=along_track_err,
            cross_track_error_m=cross_track_err,
            heading_error_rad=heading_err,
            speed_error_ms=speed_err,
            drift_percent=drift_pct
        )
        self.outage_history.append(summary)

        # 3. Closed-Loop Parameter Updates
        if duration_s >= self.min_outage_duration and distance_pred_m > 5.0:
            # (a) Gyroscope Bias Adaptation:
            # psi_integrated = psi_0 + int(w - b_g) dt
            # If heading_gnss < heading_pred (heading_err < 0), psi_integrated had positive drift,
            # meaning w_yaw was too positive, so gyro bias is positive (delta_gyro_bias > 0).
            delta_gyro_bias = -self.gyro_lr * (heading_err / duration_s)
            self.active_gyro_bias_rad_s += delta_gyro_bias
            self.active_gyro_bias_rad_s = float(np.clip(self.active_gyro_bias_rad_s, -0.05, 0.05))

            # (b) Speed Scale Factor Adaptation:
            # If along-track error > 0 (GNSS is ahead of prediction), speed was underestimated
            rel_distance_error = along_track_err / distance_pred_m
            speed_scale_delta = self.speed_lr * rel_distance_error
            self.active_speed_scale *= (1.0 + speed_scale_delta)
            self.active_speed_scale = float(np.clip(self.active_speed_scale, 0.85, 1.25))
        else:
            delta_gyro_bias = 0.0
            speed_scale_delta = 0.0

        calib_update = CalibrationUpdate(
            delta_gyro_bias_rad_s=delta_gyro_bias,
            new_gyro_bias_rad_s=self.active_gyro_bias_rad_s,
            speed_scale_factor=self.active_speed_scale,
            new_speed_scale=self.active_speed_scale
        )

        # 4. Initialize UI Anti-Teleport Smoothing
        self.reacquisition_time_s = exit_time_s
        self.last_pos_error_vec = pos_err_vec

        return summary, calib_update

    def get_smooth_display_pos(
        self,
        current_time_s: float,
        current_gnss_pos: np.ndarray
    ) -> np.ndarray:
        """
        Computes the visual map position at current_time_s.
        Glides smoothly from the predicted tunnel exit position onto the live GNSS track,
        preventing an instantaneous visual teleportation jump.
        """
        dt_since_reacq = current_time_s - self.reacquisition_time_s
        if dt_since_reacq < 0 or dt_since_reacq > (4.0 / self.smoothing_decay):
            return np.array(current_gnss_pos, dtype=float)

        # Exponential convergence: offset decays to 0
        decay = np.exp(-self.smoothing_decay * dt_since_reacq)
        offset = self.last_pos_error_vec * decay
        return np.array(current_gnss_pos, dtype=float) - offset

    @staticmethod
    def retrospective_smooth_trajectory(
        t_outage: np.ndarray,
        p_pred_outage: np.ndarray,
        pos_error_vec: np.ndarray
    ) -> np.ndarray:
        """
        Retrospectively corrects the logged outage trajectory (e.g. for post-trip review),
        distributing the terminal drift error smoothly backwards from entry (0 error)
        to exit (full error).
        """
        N = len(t_outage)
        if N <= 1:
            return p_pred_outage.copy()

        # Normalized progress s in [0, 1]
        progress = (t_outage - t_outage[0]) / (t_outage[-1] - t_outage[0])
        ramp = progress[:, np.newaxis]
        return p_pred_outage + ramp * pos_error_vec
