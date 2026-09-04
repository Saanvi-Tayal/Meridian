"""Vibration & Noise Pre-Filter Engine

Removes high-frequency engine harmonics, road surface roughness, and pothole impulses
from IMU measurements, and provides stationary state detection (ZUPT trigger).
Supports both causal (real-time stream) and zero-phase non-causal filtering.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy import signal


@dataclass
class FilteredIMU:
    """Container for processed and cleaned IMU telemetry."""
    acc_clean: np.ndarray          # Filtered acceleration (m/s^2)
    gyro_clean: np.ndarray         # Filtered angular velocity (rad/s)
    is_stationary: np.ndarray      # Boolean mask of detected standstill intervals
    pothole_spikes: np.ndarray     # Boolean mask of detected pothole/bump impulses


class VibrationPreFilter:
    """
    Multi-stage signal conditioning for automotive smartphone IMU:
    1. Hampel / Median filter for road pothole/bump shock impulse rejection.
    2. Butterworth Lowpass / Bandpass filter to suppress engine vibration harmonics.
    3. Moving-window variance thresholding for stationary (zero-velocity) detection.
    """

    def __init__(
        self,
        sampling_rate_hz: float = 10.0,
        cutoff_freq_hz: float = 3.5,
        filter_order: int = 4,
        hampel_window: int = 5,
        hampel_n_sigmas: float = 3.0,
        stationary_acc_var_thresh: float = 0.05,
        stationary_gyro_var_thresh: float = 0.003,
    ):
        self.fs = sampling_rate_hz
        self.fc = cutoff_freq_hz
        self.order = filter_order
        self.hampel_window = hampel_window
        self.hampel_n_sigmas = hampel_n_sigmas
        self.acc_var_thresh = stationary_acc_var_thresh
        self.gyro_var_thresh = stationary_gyro_var_thresh

        # Design Butterworth filter in Second-Order Sections (SOS) for numerical stability
        nyquist = 0.5 * self.fs
        normal_cutoff = min(self.fc / nyquist, 0.95)
        self.sos = signal.butter(self.order, normal_cutoff, btype="low", output="sos")

    def reject_impulse_shocks(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies Hampel filter (Median Absolute Deviation outlier detection)
        to detect and clip sharp shock spikes (potholes/expansion joints).
        x: Shape (N, 3)
        returns: (cleaned_x, spike_mask)
        """
        N, D = x.shape
        cleaned = x.copy()
        spike_mask = np.zeros((N, D), dtype=bool)
        k = self.hampel_window // 2

        for dim in range(D):
            series = x[:, dim]
            for i in range(N):
                start = max(0, i - k)
                end = min(N, i + k + 1)
                window = series[start:end]
                med = np.median(window)
                mad = np.median(np.abs(window - med))
                diff = np.abs(series[i] - med)
                sigma = 1.4826 * mad
                is_outlier = (diff > self.hampel_n_sigmas * sigma) if sigma > 1e-4 else (diff > 1.0)
                if is_outlier:
                    cleaned[i, dim] = med
                    spike_mask[i, dim] = True

        return cleaned, spike_mask

    def filter_engine_vibration(self, x: np.ndarray, real_time: bool = False) -> np.ndarray:
        """
        Filters high-frequency chassis & engine vibration using Butterworth SOS.
        If real_time is False, uses zero-phase sosfiltfilt (offline/bidirectional).
        If real_time is True, uses causal sosfilt (online on-device streaming).
        """
        if real_time:
            return signal.sosfilt(self.sos, x, axis=0)
        else:
            return signal.sosfiltfilt(self.sos, x, axis=0)

    def detect_stationary_states(self, acc: np.ndarray, gyro: np.ndarray, window_size: int = 7) -> np.ndarray:
        """
        Detects vehicle standstill (traffic lights, stops) via rolling window variance.
        Returns boolean array of shape (N,) where True = vehicle stopped (ZUPT valid).
        """
        N = len(acc)
        is_stationary = np.zeros(N, dtype=bool)
        k = window_size // 2

        acc_mag = np.linalg.norm(acc, axis=1)
        gyro_mag = np.linalg.norm(gyro, axis=1)

        for i in range(N):
            start = max(0, i - k)
            end = min(N, i + k + 1)
            var_a = np.var(acc_mag[start:end])
            var_w = np.var(gyro_mag[start:end])
            if var_a < self.acc_var_thresh and var_w < self.gyro_var_thresh:
                is_stationary[i] = True

        return is_stationary

    def process(self, acc: np.ndarray, gyro: np.ndarray, real_time: bool = False) -> FilteredIMU:
        """
        Runs complete pre-filtering pipeline on 3-axis accelerometer and gyroscope arrays.
        """
        # 1. Pothole / Road Shock Impulse Rejection
        acc_de_spiked, pothole_spikes = self.reject_impulse_shocks(acc)

        # 2. Engine Vibration Butterworth Filtering
        acc_clean = self.filter_engine_vibration(acc_de_spiked, real_time=real_time)
        gyro_clean = self.filter_engine_vibration(gyro, real_time=real_time)

        # 3. Stationary Detection (ZUPT Trigger)
        is_stationary = self.detect_stationary_states(acc_clean, gyro_clean)

        return FilteredIMU(
            acc_clean=acc_clean,
            gyro_clean=gyro_clean,
            is_stationary=is_stationary,
            pothole_spikes=pothole_spikes,
        )


class StationaryDetector:
    """Standalone real-time detector for Zero-Velocity Updates (ZUPT)."""

    def __init__(self, acc_threshold: float = 0.05, gyro_threshold: float = 0.003):
        self.acc_thresh = acc_threshold
        self.gyro_thresh = gyro_threshold

    def is_stopped(self, acc_window: np.ndarray, gyro_window: np.ndarray) -> bool:
        var_a = np.var(np.linalg.norm(acc_window, axis=1))
        var_w = np.var(np.linalg.norm(gyro_window, axis=1))
        return bool(var_a < self.acc_thresh and var_w < self.gyro_thresh)
