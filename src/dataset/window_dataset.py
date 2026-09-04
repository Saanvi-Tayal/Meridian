"""Windowed Dataset for IMU-to-Speed Deep Learning

Transforms continuous calibrated and filtered vehicle trips into fixed-length sliding
windows with physics-informed kinematic features for training and evaluation.
"""

from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset.loader import SynchronizedTrip
from src.calibration.aligner import InVehicleAligner
from src.filters.prefilter import VibrationPreFilter


class IMUSpeedWindowDataset(Dataset):
    """
    Slices preprocessed IMU telemetry into overlapping temporal windows.
    Input channels: (14, W)
      0-2:  Accel X_v (lateral), Y_v (longitudinal), Z_v (vertical) [m/s^2]
      3-5:  Gyro Yaw, Pitch, Roll [rad/s]
      6:    Accel magnitude [m/s^2]
      7:    Gyro magnitude [rad/s]
      8-10: Jerk d(a)/dt [m/s^3]
      11:   Centripetal interaction: a_lat * omega_yaw [m/s^2 * rad/s]
      12:   Kinetic power proxy: 0.5 * ||a||^2
      13:   Stationary / Standstill indicator (ZUPT flag: 0.0 or 1.0)
    Target:
      scalar speed v_fwd (m/s)
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        """
        features: Shape (M, 14, W)
        targets: Shape (M,)
        """
        self.mean = mean if mean is not None else np.mean(features, axis=(0, 2), keepdims=True)
        self.std = std if std is not None else (np.std(features, axis=(0, 2), keepdims=True) + 1e-6)

        # Normalize continuous features across channels
        norm_features = (features - self.mean) / self.std

        self.X = torch.tensor(norm_features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    @classmethod
    def extract_features_from_trip(
        cls,
        trip: SynchronizedTrip,
        window_len: int = 40,
        stride: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processes a single SynchronizedTrip:
        1. Calibrates phone frame into vehicle body frame.
        2. Applies vibration pre-filtering.
        3. Computes physics-informed kinematic features (centripetal, kinetic power, ZUPT flag).
        4. Extracts sliding windows.
        """
        # Step 1: In-Vehicle Alignment
        aligner = InVehicleAligner(min_accel_dynamics_thresh=0.4)
        calib_len = min(600, len(trip.time_s))
        calib_res = aligner.calibrate(
            acc=trip.acc[:calib_len],
            gravity=trip.gravity[:calib_len],
            speed_ref=trip.gt_speed_ms[:calib_len]
        )

        acc_veh = aligner.transform_vector(trip.acc)
        gyro_veh = aligner.transform_vector(trip.gyro)
        acc_veh[:, 2] -= calib_res.gravity_norm # Remove gravity

        # Step 2: Vibration & Noise Pre-Filter
        prefilter = VibrationPreFilter(sampling_rate_hz=1.0 / trip.dt)
        filtered = prefilter.process(acc_veh, gyro_veh, real_time=False)

        acc_clean = filtered.acc_clean
        gyro_clean = filtered.gyro_clean

        # Step 3: Derived Kinematic Features
        acc_mag = np.linalg.norm(acc_clean, axis=1, keepdims=True)
        gyro_mag = np.linalg.norm(gyro_clean, axis=1, keepdims=True)

        # Jerk (da/dt)
        dt = trip.dt if trip.dt > 0 else 0.1
        jerk = np.diff(acc_clean, axis=0, prepend=acc_clean[:1]) / dt

        # Physics-Informed Feature 1: Centripetal interaction (a_lat * omega_yaw)
        # Note: In turns, a_lat = v * omega_yaw -> (a_lat * omega_yaw) is proportional to v * omega^2
        centripetal = acc_clean[:, 0:1] * gyro_clean[:, 0:1]

        # Physics-Informed Feature 2: Kinetic energy / acceleration power proxy
        kinetic_power = 0.5 * (acc_mag ** 2)

        # Physics-Informed Feature 3: Stationary Standstill Indicator (0.0 or 1.0)
        stat_flag = filtered.is_stationary[:, None].astype(float)

        # Assemble full 14-channel feature matrix (N, 14)
        raw_channels = np.hstack([
            acc_clean,        # cols 0, 1, 2:   acc_x (lat), acc_y (long), acc_z (vert)
            gyro_clean,       # cols 3, 4, 5:   gyro_yaw, gyro_pitch, gyro_roll
            acc_mag,          # col 6:          acc_norm
            gyro_mag,         # col 7:          gyro_norm
            jerk,             # cols 8, 9, 10:  jerk_x, jerk_y, jerk_z
            centripetal,      # col 11:         a_lat * omega_yaw
            kinetic_power,    # col 12:         0.5 * ||a||^2
            stat_flag,        # col 13:         is_stationary (ZUPT trigger)
        ])

        # Step 4: Sliding Window Extraction
        N = len(raw_channels)
        windows = []
        targets = []

        for end_idx in range(window_len, N, stride):
            start_idx = end_idx - window_len
            # Transpose to (14, W) for Conv1d
            w = raw_channels[start_idx:end_idx].T
            windows.append(w)
            targets.append(trip.gt_speed_ms[end_idx - 1])

        if not windows:
            return np.empty((0, 14, window_len)), np.empty((0,))

        return np.array(windows, dtype=np.float32), np.array(targets, dtype=np.float32)

    @classmethod
    def from_trips(
        cls,
        trips: List[SynchronizedTrip],
        window_len: int = 40,
        stride: int = 2,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ) -> "IMUSpeedWindowDataset":
        """Builds dataset from a list of SynchronizedTrips."""
        all_X = []
        all_y = []
        for t in trips:
            X_t, y_t = cls.extract_features_from_trip(t, window_len=window_len, stride=stride)
            if len(X_t) > 0:
                all_X.append(X_t)
                all_y.append(y_t)

        X_all = np.concatenate(all_X, axis=0)
        y_all = np.concatenate(all_y, axis=0)

        return cls(features=X_all, targets=y_all, mean=mean, std=std)
