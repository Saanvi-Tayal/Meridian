"""Intelligent Dead Reckoning (IDR) & Sensor Fusion Engine

Combines:
1. In-Vehicle Alignment & Dynamic Calibration
2. Vibration & Noise Pre-Filtering
3. Hybrid AI Speed Estimator (SpeedNet 1D-CNN+BiGRU + Road Texture Power + ZUPT Gating)
4. Non-Holonomic Kinematic Constraints (NHC) & Heading Integration
5. GNSS Reacquisition Adaptive Calibration & Error-Feedback (Closed-loop bias/scale learning)
6. Anti-teleport UI trajectory smoothing
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import torch

from src.calibration.aligner import InVehicleAligner, CalibrationResult
from src.filters.prefilter import VibrationPreFilter, FilteredIMU
from src.models.speed_net import SpeedNet
from src.dataset.window_dataset import IMUSpeedWindowDataset
from src.fusion.reacquisition import GNSSReacquisitionManager, OutageSummary, CalibrationUpdate, wrap_angle_rad
from src.map_matching.road_network import RoadNetwork
from src.map_matching.hmm_matcher import HMMMapMatcher, MatchedState
from src.map_matching.curvature_feedback import RoadCurvatureConstraint


@dataclass
class NavigationOutput:
    """State output at 10Hz/200Hz."""
    time_s: float
    pos_east_m: float
    pos_north_m: float
    speed_kmh: float
    heading_deg: float
    is_gnss_denied: bool
    is_stationary: bool
    display_pos_east_m: float
    display_pos_north_m: float


class IntelligentDeadReckoningEngine:
    """
    Production-ready Dead Reckoning & GNSS Fusion Engine for ground vehicles.
    """

    def __init__(
        self,
        dt: float = 0.1,
        models_dir: Optional[str] = None,
        gyro_learning_rate: float = 0.5,
        speed_learning_rate: float = 0.5,
    ):
        self.dt = dt
        self.aligner = InVehicleAligner(min_accel_dynamics_thresh=0.4)
        self.prefilter = VibrationPreFilter(sampling_rate_hz=1.0 / dt)
        self.reacquisition_mgr = GNSSReacquisitionManager(
            gyro_learning_rate=gyro_learning_rate,
            speed_learning_rate=speed_learning_rate,
            smoothing_decay=2.0
        )

        if models_dir is None:
            models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
        self.models_dir = models_dir

        self.speed_model: Optional[SpeedNet] = None
        self.norm_mean: Optional[np.ndarray] = None
        self.norm_std: Optional[np.ndarray] = None
        self._load_ai_model()

        # Calibration state
        self.is_calibrated = False
        self.vib_poly = np.array([34.0, -8.0])  # Default road vibration model: v = a*rms + b

        # Navigation state
        self.is_gnss_denied = False
        self.current_pos = np.zeros(2)  # [East, North] in meters
        self.current_speed_ms = 0.0
        self.current_heading_rad = 0.0  # Radians clockwise from North
        self.outage_dist_traveled = 0.0

        # Module 6: Offline Map-Matching & Curvature Constraint
        self.map_matcher: Optional[HMMMapMatcher] = None
        self.curvature_constraint: Optional[RoadCurvatureConstraint] = None
        self.last_matched_state: Optional[MatchedState] = None

    def set_road_network(self, network: RoadNetwork, trust_curvature: bool = True):
        """Loads offline digital road network for lane-level HMM map matching."""
        self.map_matcher = HMMMapMatcher(network, sigma_z=4.0, beta=3.0)
        if trust_curvature:
            self.curvature_constraint = RoadCurvatureConstraint(trust_weight=0.35)

    def _load_ai_model(self):
        weights_path = os.path.join(self.models_dir, "speed_net.pth")
        norm_path = os.path.join(self.models_dir, "speed_net_norm.npz")
        if os.path.exists(weights_path) and os.path.exists(norm_path):
            norm_data = np.load(norm_path)
            self.norm_mean = norm_data["mean"]
            self.norm_std = norm_data["std"]

            self.speed_model = SpeedNet(in_channels=14, hidden_dim=48, num_gru_layers=2)
            self.speed_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            self.speed_model.eval()

    def calibrate_pre_outage(
        self,
        v_rms_pre: np.ndarray,
        v_gnss_pre: np.ndarray
    ):
        """Fits dynamic road-surface vibration power to GNSS speed right before outage."""
        if len(v_rms_pre) > 10 and np.std(v_rms_pre) > 1e-4:
            self.vib_poly = np.polyfit(v_rms_pre, v_gnss_pre, deg=1)
            self.vib_poly[0] = max(10.0, self.vib_poly[0])  # Enforce positive slope

    def on_gnss_lost(
        self,
        time_s: float,
        p_entry: np.ndarray,
        v_entry: float,
        heading_entry_rad: float
    ):
        """Notifies engine of GNSS drop. Initializes dead-reckoning state."""
        self.is_gnss_denied = True
        self.current_pos = np.array(p_entry, dtype=float)
        self.current_speed_ms = float(v_entry)
        self.current_heading_rad = float(heading_entry_rad)
        self.outage_dist_traveled = 0.0
        self.reacquisition_mgr.notify_gnss_lost(time_s, p_entry, heading_entry_rad)

    def on_gnss_restored(
        self,
        time_s: float,
        p_gnss: np.ndarray,
        v_gnss: float,
        heading_gnss_rad: float
    ) -> Tuple[OutageSummary, CalibrationUpdate]:
        """
        Notifies engine of GNSS reacquisition. Compares predicted exit state
        with GNSS fix, adapts calibration parameters, and resumes GNSS tracking.
        """
        summary, calib = self.reacquisition_mgr.notify_gnss_restored(
            exit_time_s=time_s,
            p_pred_exit=self.current_pos,
            v_pred_exit=self.current_speed_ms,
            heading_pred_exit_rad=self.current_heading_rad,
            distance_pred_m=self.outage_dist_traveled,
            p_gnss_exit=p_gnss,
            v_gnss_exit=v_gnss,
            heading_gnss_exit_rad=heading_gnss_rad
        )
        self.is_gnss_denied = False
        self.current_pos = np.array(p_gnss, dtype=float)
        self.current_speed_ms = float(v_gnss)
        self.current_heading_rad = float(heading_gnss_rad)
        return summary, calib

    def estimate_speed(
        self,
        features_window: Optional[np.ndarray],
        v_rms_instant: float,
        is_stationary: bool
    ) -> float:
        """
        Infers forward velocity fusing deep learning, road texture vibration, and ZUPT,
        scaled by the adaptively calibrated speed scale factor.
        """
        if is_stationary:
            return 0.0

        # Component 1: Road texture vibration power
        v_vib = max(0.0, float(np.polyval(self.vib_poly, v_rms_instant)))

        # Component 2: SpeedNet neural network
        if features_window is not None and self.speed_model is not None and self.norm_mean is not None:
            norm_feat = (features_window - self.norm_mean) / self.norm_std
            with torch.no_grad():
                tensor_in = torch.tensor(norm_feat, dtype=torch.float32).unsqueeze(0)
                v_deep = float(self.speed_model(tensor_in).item())
        else:
            v_deep = v_vib

        # Hybrid fusion with adaptive scale factor feedback
        v_raw = 0.5 * v_deep + 0.5 * v_vib
        v_scaled = v_raw * self.reacquisition_mgr.active_speed_scale
        return max(0.0, v_scaled)

    def step_dead_reckoning(
        self,
        yaw_rate_rad_s: float,
        features_window: Optional[np.ndarray],
        v_rms_instant: float,
        is_stationary: bool,
        time_s: float = 0.0
    ) -> Tuple[np.ndarray, float, float]:
        """
        Executes one dead-reckoning update step during GNSS blackout.
        Returns: (pos_east_north, speed_ms, heading_rad)
        """
        # 1. Update heading with active gyro bias compensation
        yaw_rate_corrected = yaw_rate_rad_s - self.reacquisition_mgr.active_gyro_bias_rad_s

        # Apply curvature constraint feedback if matched to road with high confidence
        if self.curvature_constraint is not None and self.last_matched_state is not None:
            seg = self.map_matcher.network.segments.get(self.last_matched_state.matched_segment_id)
            if seg is not None:
                yaw_rate_corrected, _ = self.curvature_constraint.constrain_yaw_rate(
                    measured_yaw_rate_rad_s=yaw_rate_corrected,
                    forward_speed_ms=self.current_speed_ms,
                    segment=seg,
                    fraction_s=self.last_matched_state.fraction_s,
                    is_high_confidence=(self.last_matched_state.lateral_offset_m < 8.0)
                )

        self.current_heading_rad = wrap_angle_rad(self.current_heading_rad + yaw_rate_corrected * self.dt)

        # 2. Update speed with adaptive scale factor and ZUPT
        v_est = self.estimate_speed(features_window, v_rms_instant, is_stationary)
        self.current_speed_ms = v_est

        # 3. Position dead reckoning (NHC: vehicle travels forward along heading)
        delta_dist = v_est * self.dt
        self.outage_dist_traveled += delta_dist

        delta_east = delta_dist * np.sin(self.current_heading_rad)
        delta_north = delta_dist * np.cos(self.current_heading_rad)
        self.current_pos[0] += delta_east
        self.current_pos[1] += delta_north

        # 4. Step Module 6 HMM Map Matcher (if configured)
        if self.map_matcher is not None:
            self.last_matched_state = self.map_matcher.step(
                time_s=time_s,
                pred_pos=self.current_pos,
                vehicle_heading_rad=self.current_heading_rad,
                delta_dist_dr=delta_dist
            )

        return self.current_pos.copy(), self.current_speed_ms, self.current_heading_rad

    def get_lane_matched_position(self) -> np.ndarray:
        """Returns lane-snapped position from HMM map matcher, or raw position if unmatched."""
        if self.last_matched_state is not None:
            return self.last_matched_state.snapped_pos.copy()
        return self.current_pos.copy()

    def get_display_position(self, time_s: float, raw_gnss_pos: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns position for UI rendering. If GNSS is active, applies smooth
        exponential convergence to eliminate post-tunnel teleportation jumps.
        """
        if self.is_gnss_denied or raw_gnss_pos is None:
            return self.current_pos.copy()
        return self.reacquisition_mgr.get_smooth_display_pos(time_s, raw_gnss_pos)
